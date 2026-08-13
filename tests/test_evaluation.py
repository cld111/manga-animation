"""Tests for src/manga_animation/evaluation/ -- the Phase 3.3 evaluation harness. All fixtures

are plain, hand-built `PageRunOutcome`/`EvalSample` records (no torch, no real model calls,
matching every other stage's fake-client test style) so metric arithmetic and denominator
correctness can be checked exactly.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from manga_animation.evaluation.dataset import (
    GOLDEN_DATASET_CATEGORIES,
    EvalSample,
    golden_category_coverage,
    load_eval_dataset,
    uncovered_golden_categories,
)
from manga_animation.evaluation.metrics import (
    Rate,
    StatusBreakdown,
    classify_outcome,
    compute_metrics,
)
from manga_animation.evaluation.nondeterminism import RepeatedRunRecord, summarize_repeated_runs
from manga_animation.evaluation.schemas import (
    LoopMetricsOutcome,
    ObjectAttemptOutcome,
    PageRunOutcome,
    RenderSummary,
    ValidationAttemptOutcome,
)

# --- Rate ---------------------------------------------------------------------------------


def test_rate_formats_numerator_denominator_and_percentage():
    assert str(Rate(6, 10)) == "6/10 (60.0%)"


def test_rate_zero_denominator_is_not_a_percentage():
    r = Rate(0, 0)
    assert r.value is None
    assert str(r) == "0/0 (n/a)"


def test_rate_full_and_empty():
    assert Rate(10, 10).value == pytest.approx(1.0)
    assert Rate(0, 10).value == pytest.approx(0.0)


@pytest.mark.parametrize("num,den", [(-1, 5), (5, -1), (6, 5)])
def test_rate_rejects_invalid_values(num, den):
    with pytest.raises(ValueError):
        Rate(num, den)


# --- compute_metrics fixtures -----------------------------------------------------------


def _sample(sample_id: str, **overrides) -> EvalSample:
    defaults = dict(
        sample_id=sample_id,
        image_path=f"examples/{sample_id}.png",
        source_citation="test fixture",
        diversity_tag="test",
        fetch_script="scripts/fetch_sample_pages.py",
        acceptable_outcome="anything reasonable",
    )
    defaults.update(overrides)
    return EvalSample(**defaults)


def _completed(sample_id: str, mode="page", **overrides) -> PageRunOutcome:
    defaults = dict(
        sample_id=sample_id,
        analysis_mode=mode,
        status="completed",
        primary_semantic_label="hair",
        primary_motion_type="primary",
        validation_attempts=[
            ValidationAttemptOutcome(
                candidate_rank=0, accepted=True, grounding_score=0.6, reason="ok"
            )
        ],
    )
    defaults.update(overrides)
    return PageRunOutcome(**defaults)


def _failed(
    sample_id: str, stage: str, detail: str = "failure", mode="page", **overrides
) -> PageRunOutcome:
    defaults = dict(
        sample_id=sample_id,
        analysis_mode=mode,
        status="failed",
        failing_stage=stage,
        failure_detail=detail,
    )
    defaults.update(overrides)
    return PageRunOutcome(**defaults)


def test_compute_metrics_requires_at_least_one_outcome():
    with pytest.raises(ValueError):
        compute_metrics([], {})


def test_compute_metrics_rejects_mixed_analysis_modes():
    outcomes = [_completed("a", mode="page"), _completed("b", mode="panel")]
    with pytest.raises(ValueError):
        compute_metrics(outcomes, {})


def test_usable_target_rate_excludes_only_analysis_stage_failures():
    outcomes = [
        _completed("a"),
        _failed("b", stage="analysis", detail="VLM marked every object STATIC across ..."),
        _failed("c", stage="grounding", detail="no detection above threshold"),
    ]
    report = compute_metrics(outcomes, {})
    # "a" (completed) and "c" (failed at grounding, past analysis) both count as usable-target;
    # only "b" (failed AT analysis) does not.
    assert report.usable_target_rate == Rate(2, 3)


def test_static_rate_only_counts_the_genuine_all_static_detail_text():
    outcomes = [
        _failed("a", stage="analysis", detail="VLM marked every object STATIC across every ..."),
        _failed("b", stage="analysis", detail="VLM output remained invalid JSON / schema-invalid"),
        _completed("c"),
    ]
    report = compute_metrics(outcomes, {})
    assert report.static_rate == Rate(1, 3)


def test_grounding_success_rate_denominator_is_only_pages_that_reached_grounding():
    outcomes = [
        _completed("a"),
        _failed("b", stage="grounding", detail="no detection"),
        _failed("c", stage="analysis", detail="all STATIC"),  # never reached grounding at all
    ]
    report = compute_metrics(outcomes, {})
    # denominator excludes "c" (analysis failure) -- only "a"/"b" reached grounding
    assert report.grounding_success_rate == Rate(1, 2)


def test_validation_acceptance_and_rejection_rates_pool_across_pages():
    outcomes = [
        _completed(
            "a",
            validation_attempts=[
                ValidationAttemptOutcome(
                    candidate_rank=0, accepted=False, grounding_score=0.3, reason="x"
                ),
                ValidationAttemptOutcome(
                    candidate_rank=1, accepted=True, grounding_score=0.5, reason="y"
                ),
            ],
        ),
        _failed(
            "b",
            stage="validation",
            detail="all candidates rejected",
            validation_attempts=[
                ValidationAttemptOutcome(
                    candidate_rank=0, accepted=False, grounding_score=0.4, reason="z"
                ),
            ],
        ),
    ]
    report = compute_metrics(outcomes, {})
    assert report.validation_acceptance_rate == Rate(1, 3)
    assert report.validation_rejection_rate == Rate(2, 3)


def test_validation_rate_denominator_is_zero_when_no_attempts_were_made():
    outcomes = [_failed("a", stage="analysis", detail="all STATIC")]
    report = compute_metrics(outcomes, {})
    assert report.validation_acceptance_rate == Rate(0, 0)
    assert report.validation_acceptance_rate.value is None


def test_fallback_rate_and_end_to_end_completion_rate():
    outcomes = [
        _completed("a", used_fallback_plan=True),
        _completed("b"),
        _failed("c", stage="grounding", detail="x"),
    ]
    report = compute_metrics(outcomes, {})
    assert report.fallback_rate == Rate(1, 3)
    assert report.end_to_end_completion_rate == Rate(2, 3)


def test_semantic_false_positive_rate_flags_completions_on_no_target_samples():
    samples = {
        "static_page": _sample("static_page", animation_possible="no"),
        "hair_page": _sample("hair_page", animation_possible="yes"),
    }
    outcomes = [
        _completed("static_page"),  # false positive: completed where ground truth says "no"
        _completed("hair_page"),  # correct: completed where ground truth says "yes"
    ]
    report = compute_metrics(outcomes, samples)
    assert report.semantic_false_positive_rate == Rate(1, 1)
    assert report.semantic_false_negative_rate == Rate(0, 1)


def test_semantic_false_negative_rate_flags_failures_on_yes_target_samples():
    samples = {"hair_page": _sample("hair_page", animation_possible="yes")}
    outcomes = [_failed("hair_page", stage="grounding", detail="nothing found")]
    report = compute_metrics(outcomes, samples)
    assert report.semantic_false_negative_rate == Rate(1, 1)
    assert report.semantic_false_positive_rate == Rate(0, 0)  # no "no" samples at all


def test_uncertain_ground_truth_samples_are_excluded_from_fp_fn_denominators():
    samples = {
        "ambiguous": _sample(
            "ambiguous", animation_possible="uncertain", ground_truth_uncertain=True
        )
    }
    outcomes = [_completed("ambiguous")]
    report = compute_metrics(outcomes, samples)
    assert report.semantic_false_positive_rate == Rate(0, 0)
    assert report.semantic_false_negative_rate == Rate(0, 0)
    assert report.unresolved_ground_truth_count == 1


def test_ground_truth_uncertain_flag_excludes_a_sample_even_with_a_resolved_animation_possible():
    """The exclusion gate is `ground_truth_uncertain`, not `animation_possible == "uncertain"`

    (Phase 3.4 baseline cleanup fix) -- a hedged sample that carries a resolved-looking
    `animation_possible="yes"` but is still marked `ground_truth_uncertain=True` must not
    silently contaminate semantic_false_negative_rate. Before this fix, only the literal
    `animation_possible == "uncertain"` value was excluded, so this exact case would have
    counted as a known positive.
    """
    samples = {
        "hedged": _sample("hedged", animation_possible="yes", ground_truth_uncertain=True)
    }
    outcomes = [_failed("hedged", stage="analysis", detail="all STATIC")]
    report = compute_metrics(outcomes, samples)
    assert report.semantic_false_negative_rate == Rate(0, 0)
    assert report.unresolved_ground_truth_count == 1


def test_real_dataset_uncertain_samples_are_excluded_from_semantic_metrics():
    """Real-dataset-level check: of the 5 real samples, exactly the 2 marked

    `ground_truth_uncertain=True` (`sample_page_01`, `sample_page_02`) are excluded from
    semantic_false_positive_rate/semantic_false_negative_rate's denominators, regardless of
    what the (synthetic, here) predictions say.
    """
    samples = {s.sample_id: s for s in load_eval_dataset()}
    outcomes = [_completed(sample_id) for sample_id in samples]
    report = compute_metrics(outcomes, samples)
    assert report.unresolved_ground_truth_count == 2
    resolved_count = sum(1 for s in samples.values() if not s.ground_truth_uncertain)
    assert (
        report.semantic_false_positive_rate.denominator
        + report.semantic_false_negative_rate.denominator
        == resolved_count
    )


def test_samples_missing_from_ground_truth_dict_are_skipped_not_crashed():
    outcomes = [_completed("unknown_sample")]
    report = compute_metrics(outcomes, {})  # no matching EvalSample at all
    assert report.semantic_false_positive_rate == Rate(0, 0)
    assert report.semantic_false_negative_rate == Rate(0, 0)


def test_regression_violation_is_flagged_when_completed_outcome_selects_the_wrong_target():
    """ground truth: expected target = weapon; pipeline result: COMPLETED, selected = hair --

    a real, structured mismatch, not merely "it completed" (Phase 3.4 baseline cleanup fix: the
    previous implementation flagged any completion on a flagged sample, regardless of target).
    """
    samples = {
        "flagged": _sample(
            "flagged",
            regression_reference="must never accept the known-bad face crop",
            expected_target_category="weapon",
        ),
        "unflagged": _sample("unflagged"),
    }
    outcomes = [
        _completed("flagged", primary_semantic_label="hair"),
        _completed("unflagged", primary_semantic_label="hair"),
    ]
    report = compute_metrics(outcomes, samples)
    assert report.regression_samples_checked == 1  # only "flagged" carries a reference
    assert report.regression_violation_count == 1  # completed, but selected the wrong target


def test_regression_not_violated_when_completed_outcome_selects_the_expected_target():
    """ground truth: expected target = weapon; pipeline result: COMPLETED, selected = weapon --

    a genuinely correct completion must NOT be flagged as a regression (mirrors
    `phase3_action_page`'s own `acceptable_outcome`, which explicitly allows a validated ACCEPT
    on a real weapon-shaped region as a good result).
    """
    samples = {
        "flagged": _sample(
            "flagged",
            regression_reference="must never accept the known-bad face crop",
            expected_target_category="weapon",
        ),
    }
    outcomes = [_completed("flagged", primary_semantic_label="raised_weapon")]
    report = compute_metrics(outcomes, samples)
    assert report.regression_samples_checked == 1
    assert report.regression_violation_count == 0


def test_regression_cannot_be_auto_detected_without_an_expected_target_category():
    """Mirrors the real `phase3_action_page` case: a `regression_reference` exists, but

    `expected_target_category` is deliberately left unset because the dataset does not assert
    a single correct target. `_check_regression` cannot then distinguish a correct completion
    from a regression automatically, and must not guess -- it stays `False`, not `True`, so a
    real successful completion is never falsely flagged.
    """
    samples = {
        "flagged": _sample(
            "flagged", regression_reference="must never ground to the face/dialogue-box region"
        ),
    }
    outcomes = [_completed("flagged", primary_semantic_label="raised_sword")]
    report = compute_metrics(outcomes, samples)
    assert report.regression_samples_checked == 1
    assert report.regression_violation_count == 0


def test_regression_not_violated_when_the_flagged_sample_correctly_fails():
    samples = {
        "flagged": _sample(
            "flagged",
            regression_reference="must never accept the bad crop",
            expected_target_category="weapon",
        ),
    }
    outcomes = [_failed("flagged", stage="validation", detail="rejected correctly")]
    report = compute_metrics(outcomes, samples)
    assert report.regression_samples_checked == 1
    assert report.regression_violation_count == 0


def test_panel_detection_multi_panel_rate_only_reported_in_panel_mode():
    page_outcomes = [_completed("a", mode="page", panel_count=3)]
    panel_outcomes = [
        _completed("a", mode="panel", panel_count=3),
        _completed("b", mode="panel", panel_count=1),
        _failed("c", mode="panel", stage="analysis", detail="all STATIC", panel_count=None),
    ]
    page_report = compute_metrics(page_outcomes, {})
    panel_report = compute_metrics(panel_outcomes, {})

    assert page_report.panel_detection_multi_panel_rate is None
    # "a" has panel_count=3 (>=2, counts); "b" has 1 (doesn't); "c" has None -> treated as 0
    assert panel_report.panel_detection_multi_panel_rate == Rate(1, 3)


# --- Phase 7.2.1: SECONDARY/MICRO per-object reporting -------------------------------------
#
# Closes the evaluation gap ADR 0010 explicitly deferred to Phase 7 ("extending evaluation to
# report on secondary/micro objects too is real future work... at Phase 7, not here"):
# PageRunOutcome.primary_semantic_label/primary_motion_type only ever described the PRIMARY
# object -- these tests cover the new object_outcomes list and the two render-rate metrics
# computed from it.


def _object_outcome(
    motion_type: str, status: str, object_id: str = "obj_1", semantic_label: str = "cloth"
) -> ObjectAttemptOutcome:
    return ObjectAttemptOutcome(
        object_id=object_id, semantic_label=semantic_label, motion_type=motion_type, status=status
    )


def test_page_run_outcome_defaults_to_empty_object_outcomes_and_schema_version_1():
    """Every `PageRunOutcome` recorded before Phase 7.2.1 (no `object_outcomes`/
    `schema_version` key in stored JSON) must load with the pre-Phase-7.2.1 default -- an empty
    list and schema_version=1 -- not silently be treated as "this page genuinely had zero
    SECONDARY/MICRO objects" under the new schema.
    """
    outcome = _completed("a")
    assert outcome.object_outcomes == []
    assert outcome.schema_version == 1


def test_secondary_and_micro_object_render_rates_pool_across_pages():
    outcomes = [
        _completed(
            "a",
            schema_version=2,
            object_outcomes=[
                _object_outcome("secondary", "rendered", "s1"),
                _object_outcome("secondary", "dropped", "s2"),
                _object_outcome("micro", "rendered", "m1"),
            ],
        ),
        _completed(
            "b",
            schema_version=2,
            object_outcomes=[
                _object_outcome("secondary", "rendered", "s3"),
                _object_outcome("micro", "dropped", "m2"),
            ],
        ),
    ]
    report = compute_metrics(outcomes, {})
    assert report.secondary_object_render_rate == Rate(2, 3)  # s1, s3 rendered; s2 dropped
    assert report.micro_object_render_rate == Rate(1, 2)  # m1 rendered; m2 dropped


def test_secondary_object_render_rate_denominator_is_zero_with_no_object_outcomes():
    """The common, pre-Phase-7.2.1-equivalent case (a single-PRIMARY-only plan, or every
    outcome still at schema_version=1) must report "0/0 (n/a)", never a fabricated 0%.
    """
    outcomes = [_completed("a"), _failed("b", stage="grounding", detail="x")]
    report = compute_metrics(outcomes, {})
    assert report.secondary_object_render_rate == Rate(0, 0)
    assert report.secondary_object_render_rate.value is None
    assert report.micro_object_render_rate == Rate(0, 0)


def test_object_outcomes_do_not_affect_primary_only_metrics():
    """Adding object_outcomes to a PageRunOutcome must not change any pre-existing
    PRIMARY-only rate -- this is a purely additive extension.
    """
    plain = compute_metrics([_completed("a")], {})
    with_secondary = compute_metrics(
        [
            _completed(
                "a",
                schema_version=2,
                object_outcomes=[_object_outcome("secondary", "rendered")],
            )
        ],
        {},
    )
    assert plain.end_to_end_completion_rate == with_secondary.end_to_end_completion_rate
    assert plain.usable_target_rate == with_secondary.usable_target_rate
    assert plain.validation_acceptance_rate == with_secondary.validation_acceptance_rate


def test_object_attempt_outcome_round_trips_through_json():
    outcome = _completed(
        "a",
        schema_version=2,
        object_outcomes=[
            ObjectAttemptOutcome(
                object_id="obj_2",
                semantic_label="trailing_cloth",
                motion_type="secondary",
                status="dropped",
                validation_attempts=[
                    ValidationAttemptOutcome(
                        candidate_rank=0, accepted=False, grounding_score=0.4, reason="no match"
                    )
                ],
            )
        ],
    )
    restored = PageRunOutcome.model_validate_json(outcome.model_dump_json())
    assert restored.object_outcomes[0].object_id == "obj_2"
    assert restored.object_outcomes[0].status == "dropped"
    assert restored.object_outcomes[0].validation_attempts[0].reason == "no match"
    assert restored.schema_version == 2


# --- nondeterminism -----------------------------------------------------------------------


def test_summarize_repeated_runs_requires_at_least_one_record():
    with pytest.raises(ValueError):
        summarize_repeated_runs([])


def test_summarize_repeated_runs_rejects_mixed_sample_ids():
    records = [
        RepeatedRunRecord(sample_id="a", run_index=0, outcome="usable"),
        RepeatedRunRecord(sample_id="b", run_index=1, outcome="usable"),
    ]
    with pytest.raises(ValueError):
        summarize_repeated_runs(records)


def test_summarize_repeated_runs_stable_when_every_run_agrees():
    records = [
        RepeatedRunRecord(
            sample_id="s", run_index=i, outcome="usable", primary_semantic_label="hair"
        )
        for i in range(3)
    ]
    summary = summarize_repeated_runs(records)
    assert summary.outcome_stable is True
    assert summary.target_category_stable is True
    assert summary.distinct_outcomes == ["usable"]
    assert summary.distinct_primary_labels == ["hair"]


def test_summarize_repeated_runs_detects_outcome_flip():
    """Mirrors the real, documented Phase 3.2 finding: sample_page_01 flipped between an

    all-STATIC read and a usable character_hair PRIMARY read across runs.
    """
    records = [
        RepeatedRunRecord(sample_id="s", run_index=0, outcome="static_or_unusable"),
        RepeatedRunRecord(
            sample_id="s", run_index=1, outcome="usable", primary_semantic_label="character_hair"
        ),
    ]
    summary = summarize_repeated_runs(records)
    assert summary.outcome_stable is False
    assert summary.distinct_outcomes == ["static_or_unusable", "usable"]


def test_summarize_repeated_runs_detects_target_category_change():
    records = [
        RepeatedRunRecord(
            sample_id="s", run_index=0, outcome="usable", primary_semantic_label="hair"
        ),
        RepeatedRunRecord(
            sample_id="s", run_index=1, outcome="usable", primary_semantic_label="cape"
        ),
    ]
    summary = summarize_repeated_runs(records)
    assert summary.outcome_stable is True  # both runs were "usable"
    assert summary.target_category_stable is False
    assert summary.distinct_primary_labels == ["cape", "hair"]


def test_summarize_repeated_runs_all_static_is_outcome_stable_with_no_labels():
    records = [
        RepeatedRunRecord(sample_id="s", run_index=i, outcome="static_or_unusable")
        for i in range(3)
    ]
    summary = summarize_repeated_runs(records)
    assert summary.outcome_stable is True
    assert summary.target_category_stable is True  # vacuously -- no usable runs to disagree
    assert summary.distinct_primary_labels == []


# --- dataset ------------------------------------------------------------------------------


def test_real_eval_dataset_manifest_loads_and_validates():
    samples = load_eval_dataset()
    assert len(samples) >= 3
    ids = [s.sample_id for s in samples]
    assert len(ids) == len(set(ids))  # no duplicate sample_ids
    for sample in samples:
        assert sample.animation_possible in ("yes", "no", "uncertain")
        assert sample.acceptable_outcome  # never empty


def test_eval_dataset_never_fabricates_expected_region_as_a_bbox():
    """Ground-truth region info must stay a qualitative note, never a pixel bbox -- no sample

    has an independently measured ground-truth region (see EvalSample's docstring)."""
    samples = load_eval_dataset()
    for sample in samples:
        if sample.expected_region_note is not None:
            assert isinstance(sample.expected_region_note, str)


def test_eval_dataset_uncertain_samples_are_explicitly_flagged():
    samples = load_eval_dataset()
    uncertain = [s for s in samples if s.animation_possible == "uncertain"]
    for sample in uncertain:
        assert sample.ground_truth_uncertain is True


# --- Phase 8: golden E2E dataset category coverage -----------------------------------------


def test_eval_sample_defaults_to_no_golden_categories():
    sample = _sample("a")
    assert sample.golden_categories == []


def test_golden_category_coverage_maps_every_required_category():
    samples = [
        _sample("a", golden_categories=["single_animatable_object", "translation"]),
        _sample("b", golden_categories=["translation"]),
    ]
    coverage = golden_category_coverage(samples)
    assert set(coverage.keys()) == set(GOLDEN_DATASET_CATEGORIES)
    assert coverage["translation"] == ["a", "b"]
    assert coverage["single_animatable_object"] == ["a"]
    assert coverage["rotation"] == []  # a real, honest gap -- empty, not absent


def test_uncovered_golden_categories_lists_only_empty_ones():
    samples = [_sample("a", golden_categories=["translation"])]
    uncovered = uncovered_golden_categories(samples)
    assert "translation" not in uncovered
    assert "rotation" in uncovered
    assert set(uncovered) == set(GOLDEN_DATASET_CATEGORIES) - {"translation"}


def test_golden_category_coverage_rejects_an_unknown_category_at_load_time():
    with pytest.raises(ValidationError):
        _sample("a", golden_categories=["not_a_real_category"])


def test_real_golden_dataset_has_exactly_the_two_disclosed_coverage_gaps():
    """The real dataset's own header comment (configs/phase3_3_eval_dataset.yaml) discloses

    two categories with zero real coverage -- locks that honest disclosure in as a checkable
    fact instead of only prose, and would fail loudly if a future edit silently covered (or
    silently dropped coverage for) one of these without updating the header note.
    """
    samples = load_eval_dataset()
    assert set(uncovered_golden_categories(samples)) == {
        "partially_occluded_object",
        "scale_or_deformation",
    }


def test_real_golden_dataset_every_sample_has_at_least_one_category():
    samples = load_eval_dataset()
    for sample in samples:
        assert sample.golden_categories, f"{sample.sample_id} has no golden_categories"


def test_eval_sample_loader_roundtrips_a_minimal_yaml(tmp_path):
    manifest = tmp_path / "mini.yaml"
    manifest.write_text(
        """
samples:
  - sample_id: mini
    image_path: examples/mini.png
    source_citation: test
    diversity_tag: test
    fetch_script: scripts/fetch_sample_pages.py
    acceptable_outcome: anything
"""
    )
    samples = load_eval_dataset(manifest)
    assert len(samples) == 1
    assert samples[0].animation_possible == "uncertain"  # the schema's own safe default
    assert samples[0].expected_target_category is None
    assert samples[0].annotation_version == 1  # the schema's own safe default


# --- ground-truth integrity (Phase 3.3.2) --------------------------------------------------
#
# Regression tests for the evaluation-oracle-instability failure class: a real, evidenced
# incident where `sample_page_02`'s ground truth (`animation_possible: "yes"`) had originally
# been set on the strength of a single VLM read, then two further independent real sessions had
# the same VLM read the same page as all-STATIC (docs/phase3.3-results.md's "VLM
# nondeterminism" section; docs/decisions/0009-evaluation-ground-truth-integrity.md). These
# tests protect the fix: VLM output must never be able to define or silently change stored
# ground truth, and evaluation must always compare a prediction against ground truth, never
# treat one as the other.


def test_eval_sample_ground_truth_is_frozen():
    """A loaded `EvalSample` cannot be mutated in place -- not by a VLM call, not by any other

    in-process code. This is the direct fix for the failure mode this phase investigated: ground
    truth silently drifting to match whatever a VLM said most recently.
    """
    sample = _sample("s", animation_possible="yes")
    with pytest.raises(ValidationError):
        sample.animation_possible = "no"  # type: ignore[misc]
    assert sample.animation_possible == "yes"  # the mutation attempt did not partially apply


def test_eval_sample_construction_still_works_when_frozen():
    """Frozen only blocks post-construction mutation -- normal construction (how every loader

    and test fixture builds a sample) is unaffected.
    """
    sample = _sample("s", animation_possible="no", ground_truth_uncertain=False)
    assert sample.animation_possible == "no"


def test_compute_metrics_result_depends_only_on_stored_ground_truth_not_on_predictions():
    """Two directly conflicting predictions (`PageRunOutcome`s) for the same sample_id, scored

    against the SAME stored ground truth, must each be judged against that one fixed ground
    truth -- never against each other, and never by treating either prediction as if it were
    itself the ground truth.
    """
    samples = {"hair_page": _sample("hair_page", animation_possible="yes")}

    vlm_session_a_said_usable = compute_metrics([_completed("hair_page")], samples)
    vlm_session_b_said_static = compute_metrics(
        [_failed("hair_page", stage="analysis", detail="VLM marked every object STATIC")],
        samples,
    )

    # Same ground truth, opposite predictions -> opposite semantic-false-negative verdicts;
    # the ground truth sample itself is untouched by either call.
    assert vlm_session_a_said_usable.semantic_false_negative_rate == Rate(0, 1)
    assert vlm_session_b_said_static.semantic_false_negative_rate == Rate(1, 1)
    assert samples["hair_page"].animation_possible == "yes"


def test_repeated_evaluation_never_mutates_the_real_dataset_manifest():
    """Running evaluation (even with wildly different, conflicting predictions) many times must

    never alter the on-disk manifest -- ground truth changes only ever happen by hand-editing
    `configs/phase3_3_eval_dataset.yaml` and committing the change.
    """
    from manga_animation.evaluation.dataset import DEFAULT_DATASET_PATH

    before = DEFAULT_DATASET_PATH.read_bytes()

    samples = load_eval_dataset()
    samples_by_id = {s.sample_id: s for s in samples}
    conflicting_predictions = [
        _completed(s.sample_id) if i % 2 == 0 else _failed(s.sample_id, stage="analysis")
        for i, s in enumerate(samples)
    ]
    for _ in range(5):
        compute_metrics(conflicting_predictions, samples_by_id)
        load_eval_dataset()  # reload too -- must not pick up any in-memory drift

    after = DEFAULT_DATASET_PATH.read_bytes()
    assert after == before


def test_real_dataset_ground_truth_changes_carry_an_explicit_annotation_version():
    """`sample_page_02` is this project's original real, evidenced case of a ground-truth

    revision -- its `annotation_version` must reflect that it was intentionally revised (2).
    Phase 8.3 added a second, real revision reason (`honest_failure_acceptable`, formalizing
    `phase3_action_page`/`eval_weapon_effects`'s own pre-existing `acceptable_outcome` prose
    into a structured field) -- those two also sit at version 2. Every other, unrevised sample
    stays at the schema's default (1). A version bump is the auditable signal that a human
    reviewed and changed the annotation, not that a VLM run overwrote it.
    """
    samples = {s.sample_id: s for s in load_eval_dataset()}
    assert samples["sample_page_02"].annotation_version == 2
    assert samples["sample_page_02"].animation_possible == "uncertain"
    assert samples["sample_page_02"].ground_truth_uncertain is True

    revised_at_v2 = {"sample_page_02", "phase3_action_page", "eval_weapon_effects"}
    for sample_id, sample in samples.items():
        expected = 2 if sample_id in revised_at_v2 else 1
        assert sample.annotation_version == expected, sample_id


def test_transform_geometry_failure_does_not_alter_semantic_ground_truth():
    """A sample can be semantically true-positive (`animation_possible="yes"`) while a specific

    run fails at the transform-geometry-validation stage (Phase 3.3.1/ADR 0008) -- that failure
    must count against the pipeline's semantic-false-negative rate exactly like any other
    failure, but must never be mistaken for, or change, the sample's stored semantic ground
    truth. Guards against collapsing "semantically animatable" and "safe for this transform"
    into one label (see ADR 0009's explicit distinction from ADR 0008's).
    """
    samples = {"weapon_page": _sample("weapon_page", animation_possible="yes")}
    geometrically_unsafe_outcome = _failed(
        "weapon_page",
        stage="validation",
        detail=(
            "all 1 grounding candidate(s) for semantic_label='weapon' failed target "
            "validation: rank=0 bbox covers 27.6% of its reference region, exceeding the 15% "
            "bound a rotate target allows"
        ),
    )
    report = compute_metrics([geometrically_unsafe_outcome], samples)
    assert report.semantic_false_negative_rate == Rate(1, 1)
    # the ground truth itself never changed -- still semantically "yes", not downgraded because
    # this particular candidate was geometrically unsafe for its transform.
    assert samples["weapon_page"].animation_possible == "yes"


def test_compute_metrics_is_a_pure_deterministic_function_of_its_inputs():
    """Repeated evaluation over identical predictions and identical ground truth must produce

    an identical `EvaluationReport` -- the comparison procedure itself is deterministic, even
    though the real VLM predictions it consumes are not.
    """
    samples = {
        "a": _sample("a", animation_possible="yes"),
        "b": _sample("b", animation_possible="no"),
    }
    outcomes = [_completed("a"), _failed("b", stage="analysis", detail="all STATIC")]

    reports = [compute_metrics(outcomes, samples) for _ in range(5)]
    assert all(r == reports[0] for r in reports)


# --- verified-action dataset integration (Pre-Phase-3.4) --------------------------------------


def test_verified_action_samples_are_real_frozen_immutable_positive_controls():
    """The two real `examples/verified_action/` samples load as ordinary, frozen `EvalSample`s

    with a resolved, non-uncertain positive label -- verified positive controls are ground
    truth like any other, not a second parallel representation.
    """
    samples = {s.sample_id: s for s in load_eval_dataset()}
    for sample_id in ("verified_action_1", "verified_action_2"):
        sample = samples[sample_id]
        assert sample.animation_possible == "yes"
        assert sample.ground_truth_uncertain is False
        with pytest.raises(ValidationError):
            sample.animation_possible = "no"  # type: ignore[misc]


def test_verified_action_provenance_is_independent_human_verification_not_vlm():
    """Provenance is recorded structurally, and is never the VLM/pipeline's own output --

    `PageRunOutcome` (a prediction) has no `annotation_provenance`-shaped field at all, so there
    is no code path through which a pipeline run could even resemble writing one.
    """
    samples = {s.sample_id: s for s in load_eval_dataset()}
    for sample_id in ("verified_action_1", "verified_action_2"):
        assert samples[sample_id].annotation_provenance == "independent_human_verification"
    assert not hasattr(PageRunOutcome, "annotation_provenance")


def test_verified_action_provenance_is_unaffected_by_any_prediction():
    """Mirrors `test_compute_metrics_result_depends_only_on_stored_ground_truth_not_on_predictions`

    specifically for the new provenance field: wildly different predictions for the same
    verified sample must never change its stored provenance or resolved status.
    """
    samples = {s.sample_id: s for s in load_eval_dataset()}
    sample = samples["verified_action_1"]

    compute_metrics([_completed("verified_action_1")], samples)
    compute_metrics(
        [_failed("verified_action_1", stage="analysis", detail="VLM marked every object STATIC")],
        samples,
    )
    assert sample.annotation_provenance == "independent_human_verification"
    assert sample.animation_possible == "yes"
    assert sample.ground_truth_uncertain is False


def test_real_dataset_ground_truth_split_is_visible_and_sums_to_sample_count():
    """The 3-way split Pre-Phase-3.4 requires (verified positive / verified negative /

    unresolved) is derivable from the existing `EvaluationReport` fields without a second
    parallel metric system, and covers every sample exactly once.
    """
    samples = {s.sample_id: s for s in load_eval_dataset()}
    outcomes = [_completed(sample_id) for sample_id in samples]
    report = compute_metrics(outcomes, samples)

    positive_controls = report.semantic_false_negative_rate.denominator
    negative_controls = report.semantic_false_positive_rate.denominator
    unresolved = report.unresolved_ground_truth_count
    assert positive_controls + negative_controls + unresolved == report.sample_count == len(
        samples
    )
    # verified_action_1/2 count as resolved positive controls, not unresolved.
    assert positive_controls >= 2


def test_adding_verified_action_samples_did_not_mutate_the_pre_existing_annotations():
    """Regression guard: the two new manifest entries must not have altered any of the

    5 pre-existing samples' stored ground truth -- an addition, not a rewrite.
    """
    samples = {s.sample_id: s for s in load_eval_dataset()}
    assert samples["sample_page_01"].animation_possible == "uncertain"
    assert samples["sample_page_01"].ground_truth_uncertain is True
    assert samples["sample_page_01"].annotation_version == 1
    assert samples["sample_page_02"].animation_possible == "uncertain"
    assert samples["sample_page_02"].ground_truth_uncertain is True
    assert samples["sample_page_02"].annotation_version == 2
    assert samples["phase3_action_page"].animation_possible == "yes"
    assert samples["eval_static_dialogue"].animation_possible == "no"
    assert samples["eval_weapon_effects"].animation_possible == "yes"


def test_load_eval_dataset_rejects_duplicate_image_path(tmp_path):
    """The same image must never carry two conflicting ground-truth identities -- a copy-pasted

    manifest entry pointing at an already-used image_path is rejected at load time, not
    silently accepted with two disagreeing sample_ids for one picture.
    """
    manifest = tmp_path / "dupes.yaml"
    manifest.write_text(
        """
samples:
  - sample_id: first
    image_path: examples/shared.png
    source_citation: test
    diversity_tag: test
    acceptable_outcome: anything
    animation_possible: "yes"
  - sample_id: second
    image_path: examples/shared.png
    source_citation: test
    diversity_tag: test
    acceptable_outcome: anything
    animation_possible: "no"
"""
    )
    with pytest.raises(ValueError, match="duplicate image_path"):
        load_eval_dataset(manifest)


def test_verified_action_samples_do_not_invent_target_or_region_ground_truth():
    """Independent verification of action presence establishes ONLY animation_possible="yes" --

    it must not be treated as if it also established a specific target/region/transform, which
    would require separate independent evidence this integration never had.
    """
    samples = {s.sample_id: s for s in load_eval_dataset()}
    for sample_id in ("verified_action_1", "verified_action_2"):
        sample = samples[sample_id]
        assert sample.expected_target_category is None
        assert sample.expected_motion_category is None
        assert sample.expected_region_note is None


# --- Phase 8: E2E status classification (PASS / PASS_WITH_FALLBACK / REJECTED / ERROR) -----


def _render_summary(**overrides) -> RenderSummary:
    defaults = dict(
        frame_count=96,
        fps=24.0,
        resolution=(720, 1000),
        duration_s=4.0,
        codec="h264",
        pixel_format="yuv420p",
        seamless_loop_verified=True,
    )
    defaults.update(overrides)
    return RenderSummary(**defaults)


def test_classify_outcome_pass_for_a_fully_automatic_completion():
    outcome = _completed("a")
    assert classify_outcome(outcome, _sample("a")) == "PASS"


def test_classify_outcome_pass_with_fallback_when_a_controlled_plan_was_used():
    outcome = _completed("a", used_fallback_plan=True)
    assert classify_outcome(outcome, _sample("a")) == "PASS_WITH_FALLBACK"


def test_classify_outcome_rejected_for_an_attributed_failure_with_no_ground_truth_conflict():
    outcome = _failed("a", stage="analysis", detail="every object STATIC")
    assert classify_outcome(outcome, _sample("a", animation_possible="uncertain")) == "REJECTED"


def test_classify_outcome_rejected_for_a_harness_attributed_unexpected_exception():
    """`failing_stage="unexpected"` (the evaluation harness's own catch-all attribution, see

    `evaluation.schemas.FailingStage`) is still an attributed reason -- REJECTED, not ERROR.
    """
    outcome = _failed("a", stage="unexpected", detail="RuntimeError: boom")
    assert classify_outcome(outcome, None) == "REJECTED"


def test_classify_outcome_error_for_a_completely_unattributed_failure():
    outcome = PageRunOutcome(
        sample_id="a", analysis_mode="page", status="failed", failing_stage=None
    )
    assert classify_outcome(outcome, None) == "ERROR"


def test_classify_outcome_error_for_a_semantic_false_positive():
    sample = _sample("a", animation_possible="no")
    outcome = _completed("a")
    assert classify_outcome(outcome, sample) == "ERROR"


def test_classify_outcome_error_for_a_semantic_false_negative():
    sample = _sample("a", animation_possible="yes")
    outcome = _failed("a", stage="analysis", detail="every object STATIC")
    assert classify_outcome(outcome, sample) == "ERROR"


def test_classify_outcome_error_for_a_reproduced_regression():
    sample = _sample(
        "a",
        regression_reference="must never accept the bad crop",
        expected_target_category="weapon",
    )
    outcome = _completed("a", primary_semantic_label="hair")
    assert classify_outcome(outcome, sample) == "ERROR"


def test_classify_outcome_uncertain_ground_truth_never_triggers_a_semantic_error():
    """A confident-looking "yes"/"no" on a sample explicitly marked `ground_truth_uncertain`

    must not be used to flag ERROR -- mirrors compute_metrics's own fp/fn exclusion.
    """
    sample = _sample("a", animation_possible="no", ground_truth_uncertain=True)
    outcome = _completed("a")
    assert classify_outcome(outcome, sample) == "PASS"


def test_classify_outcome_handles_a_missing_sample_gracefully():
    outcome = _completed("a")
    assert classify_outcome(outcome, None) == "PASS"


def test_classify_outcome_rejected_for_an_honest_attributed_failure_when_sample_allows_it():
    """Phase 8.3: a real, previously-observed mismatch -- `eval_weapon_effects`/

    `phase3_action_page`'s own `acceptable_outcome` prose has always allowed an honest
    grounding/validation failure despite confident animation_possible="yes", but
    `classify_outcome` only consulted structured fields and classified this as ERROR on real
    Kaggle GPU output (docs/phase8-results.md section 6.2). `honest_failure_acceptable` closes
    that gap: an attributed failure on such a sample is REJECTED (an honest negative), not
    ERROR.
    """
    sample = _sample("a", animation_possible="yes", honest_failure_acceptable=True)
    outcome = _failed("a", stage="validation", detail="all candidates failed target validation")
    assert classify_outcome(outcome, sample) == "REJECTED"


def test_classify_outcome_still_errors_on_unattributed_failure_despite_honest_failure_acceptable():
    """`honest_failure_acceptable` only excuses an *attributed* failure -- a genuinely

    unattributed one (`failing_stage=None`) is never "honest" and must still be ERROR.
    """
    sample = _sample("a", animation_possible="yes", honest_failure_acceptable=True)
    outcome = PageRunOutcome(
        sample_id="a", analysis_mode="page", status="failed", failing_stage=None
    )
    assert classify_outcome(outcome, sample) == "ERROR"


def test_classify_outcome_still_errors_on_confident_yes_failure_without_the_opt_in():
    """Sanity check: `honest_failure_acceptable` defaults to False, so every sample that has

    never opted in (e.g. `verified_action_1`/`verified_action_2`, whose own acceptable_outcome
    explicitly treats any failure as a false negative) keeps the pre-Phase-8.3 ERROR behavior.
    """
    sample = _sample("a", animation_possible="yes")
    outcome = _failed("a", stage="validation", detail="all candidates failed target validation")
    assert classify_outcome(outcome, sample) == "ERROR"


def test_real_dataset_honest_failure_acceptable_matches_documented_samples():
    """Locks in exactly which real samples opted into `honest_failure_acceptable`, and that

    every one of them has prose in `acceptable_outcome` that actually says so -- catches a
    future edit that flips the flag without updating the sample's own written contract, or
    vice versa.
    """
    samples = {s.sample_id: s for s in load_eval_dataset()}
    opted_in = {sid for sid, s in samples.items() if s.honest_failure_acceptable}
    assert opted_in == {"phase3_action_page", "eval_weapon_effects"}
    for sample_id in opted_in:
        prose = samples[sample_id].acceptable_outcome.lower()
        assert "honest" in prose
        assert "acceptable" in prose


def test_status_breakdown_rejects_negative_counts():
    with pytest.raises(ValueError):
        StatusBreakdown(-1, 0, 0, 0)


def test_status_breakdown_total_sums_all_four_buckets():
    breakdown = StatusBreakdown(
        pass_count=2, pass_with_fallback_count=1, rejected_count=3, error_count=1
    )
    assert breakdown.total == 7


def test_compute_metrics_status_breakdown_sums_to_sample_count():
    samples = {
        "pass": _sample("pass"),
        "fallback": _sample("fallback"),
        "rejected": _sample("rejected", animation_possible="uncertain"),
        "error": _sample("error", animation_possible="no"),
    }
    outcomes = [
        _completed("pass"),
        _completed("fallback", used_fallback_plan=True),
        _failed("rejected", stage="analysis", detail="every object STATIC"),
        _completed("error"),  # semantic false positive against animation_possible="no"
    ]
    report = compute_metrics(outcomes, samples)
    assert report.status_breakdown == StatusBreakdown(
        pass_count=1, pass_with_fallback_count=1, rejected_count=1, error_count=1
    )
    assert report.status_breakdown.total == report.sample_count == 4


# --- Phase 8: RenderSummary / LoopMetricsOutcome round-tripping ----------------------------


def test_render_summary_round_trips_through_json_with_loop_metrics():
    summary = _render_summary(
        loop_metrics=LoopMetricsOutcome(
            ordinary_adjacent_step_mean_abs_diff=1.5,
            wrap_step_mean_abs_diff=1.8,
            wrap_step_within_2x_ordinary=True,
            ordinary_adjacent_step_ssim=0.98,
            wrap_step_ssim=0.97,
            wrap_ssim_within_tolerance=True,
        )
    )
    restored = RenderSummary.model_validate_json(summary.model_dump_json())
    assert restored == summary


def test_render_summary_loop_metrics_defaults_to_none():
    summary = _render_summary()
    assert summary.loop_metrics is None


def test_page_run_outcome_render_summary_defaults_to_none_and_schema_version_1():
    outcome = _failed("a", stage="analysis")
    assert outcome.render_summary is None
    assert outcome.schema_version == 1


def test_page_run_outcome_accepts_a_populated_render_summary():
    outcome = _completed("a", render_summary=_render_summary(), schema_version=3)
    assert outcome.render_summary is not None
    assert outcome.render_summary.frame_count == 96
    assert outcome.schema_version == 3
