"""Measuring (not fixing) VLM run-to-run nondeterminism -- see the Phase 3.3 brief's "VLM

NONDETERMINISM" section and the real, already-documented finding this follows up on
(`docs/phase3.2-results.md`: `Qwen25VLClient.generate()` pins no seed/temperature, so identical
input can produce a different STATIC/PRIMARY read across runs).

Deliberately scoped to what `analyze_page`/`analyze_page_panels`'s public `AnimationPlan`
output actually exposes: whether the page produced a usable (non-STATIC) result, and the
chosen PRIMARY object's semantic label/motion type. The pre-collapse ranked-candidate list
(every candidate's motion_type/confidence before one is chosen as PRIMARY and the rest are
forced to STATIC) is an internal `plan_builder` representation, not part of the public
`AnimationPlan` -- measuring finer-grained "candidate ordering" would need a new debug-only
hook into `_rank_candidates`/`_rank_panel_candidates`, which this phase deliberately does not
add (see docs/phase3.3-results.md's "Remaining limitations" for why: avoiding speculative API
surface growth for a measurement this phase's real evidence doesn't yet show is necessary).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RunOutcomeKind = Literal["usable", "static_or_unusable"]


@dataclass(frozen=True, slots=True)
class RepeatedRunRecord:
    """One repeated analysis call's observable result for one sample."""

    sample_id: str
    run_index: int
    outcome: RunOutcomeKind
    primary_semantic_label: str | None = None
    primary_motion_type: str | None = None
    object_count: int | None = None


@dataclass(frozen=True, slots=True)
class NondeterminismSummary:
    """Whether repeated runs against the identical input agreed with each other -- both

    dimensions the Phase 3.3 brief names explicitly: "STATIC remains STATIC" (`outcome_stable`)
    and "target category changes" (`target_category_stable`).
    """

    sample_id: str
    run_count: int
    outcome_stable: bool
    target_category_stable: bool
    distinct_outcomes: list[str] = field(default_factory=list)
    distinct_primary_labels: list[str] = field(default_factory=list)


def summarize_repeated_runs(records: list[RepeatedRunRecord]) -> NondeterminismSummary:
    """Raises `ValueError` on an empty or sample_id-mixed list -- this summarizes ONE sample's

    repeated runs at a time, never silently pools different samples together.
    """
    if not records:
        raise ValueError("summarize_repeated_runs requires at least one record")
    sample_ids = {r.sample_id for r in records}
    if len(sample_ids) > 1:
        raise ValueError(
            f"summarize_repeated_runs requires every record to share one sample_id, "
            f"got {sample_ids}"
        )
    sample_id = next(iter(sample_ids))

    distinct_outcomes: list[str] = sorted({r.outcome for r in records})
    usable_labels: list[str] = sorted(
        {
            r.primary_semantic_label
            for r in records
            if r.outcome == "usable" and r.primary_semantic_label
        }
    )

    return NondeterminismSummary(
        sample_id=sample_id,
        run_count=len(records),
        outcome_stable=len(distinct_outcomes) <= 1,
        target_category_stable=len(usable_labels) <= 1,
        distinct_outcomes=distinct_outcomes,
        distinct_primary_labels=usable_labels,
    )
