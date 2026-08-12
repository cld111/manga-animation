"""Tests for src/manga_animation/evaluation/ -- the Phase 3.3 evaluation harness. All fixtures

are plain, hand-built `PageRunOutcome`/`EvalSample` records (no torch, no real model calls,
matching every other stage's fake-client test style) so metric arithmetic and denominator
correctness can be checked exactly.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from manga_animation.evaluation.dataset import EvalSample, load_eval_dataset
from manga_animation.evaluation.metrics import Rate, compute_metrics
from manga_animation.evaluation.nondeterminism import RepeatedRunRecord, summarize_repeated_runs
from manga_animation.evaluation.schemas import PageRunOutcome, ValidationAttemptOutcome

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
    """`sample_page_02` is this project's one real, evidenced case of a ground-truth revision --

    its `annotation_version` must reflect that it was intentionally revised (2), while every
    unrevised sample stays at the schema's default (1). A version bump is the auditable signal
    that a human reviewed and changed the annotation, not that a VLM run overwrote it.
    """
    samples = {s.sample_id: s for s in load_eval_dataset()}
    assert samples["sample_page_02"].annotation_version == 2
    assert samples["sample_page_02"].animation_possible == "uncertain"
    assert samples["sample_page_02"].ground_truth_uncertain is True
    for sample_id, sample in samples.items():
        if sample_id != "sample_page_02":
            assert sample.annotation_version == 1


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
