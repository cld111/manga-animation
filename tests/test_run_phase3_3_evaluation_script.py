"""Regression test for scripts/run_phase3_3_evaluation.py's `_render_rates_in_place` helper.

A Phase 7 closure audit found a real bug here: the JSON-summary rate-rendering loop indexed
`reports` by a `mode` variable left over from an unrelated, already-finished loop earlier in
`main()` (which always held its last value, `"panel"`, by the time the rate-rendering loop
ran) instead of the mode it was actually rendering -- silently corrupting the page report's
"rendered" display strings in the saved JSON with the panel report's own values (the raw
`numerator`/`denominator` fields were unaffected, only the human-readable string). Extracted
into its own function specifically so this could be regression-tested without needing to run
the whole script (real models, GPU-only, not locally runnable per ADR 0003).

`scripts/` has no `__init__.py` (a deliberate, pre-existing convention -- these are one-off
driver scripts, not an importable package), so the module is loaded directly by file path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from manga_animation.evaluation.metrics import Rate

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_phase3_3_evaluation.py"
_spec = importlib.util.spec_from_file_location("run_phase3_3_evaluation", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

_render_rates_in_place = _module._render_rates_in_place
EvaluationReport = _module.EvaluationReport


def _report(sample_count: int, usable_num: int, usable_den: int) -> EvaluationReport:
    """A minimal-but-complete EvaluationReport, only `usable_target_rate` varies per call --
    every other Rate is fixed so the test can tell "page's own value" from "panel's leaked
    value" unambiguously via `usable_target_rate` alone.
    """
    return EvaluationReport(
        analysis_mode="page",
        sample_count=sample_count,
        usable_target_rate=Rate(usable_num, usable_den),
        static_rate=Rate(0, sample_count),
        grounding_success_rate=Rate(0, 0),
        validation_acceptance_rate=Rate(0, 0),
        validation_rejection_rate=Rate(0, 0),
        fallback_rate=Rate(0, sample_count),
        end_to_end_completion_rate=Rate(0, sample_count),
        semantic_false_positive_rate=Rate(0, 0),
        semantic_false_negative_rate=Rate(0, 0),
        unresolved_ground_truth_count=0,
        regression_violation_count=0,
        regression_samples_checked=0,
        panel_detection_multi_panel_rate=None,
        secondary_object_render_rate=Rate(0, 0),
        micro_object_render_rate=Rate(0, 0),
    )


def test_render_rates_in_place_uses_each_modes_own_report_not_a_leaked_last_mode():
    """The real bug this test protects: page and panel reports must each render their OWN

    Rate strings. Deliberately gives page and panel very different usable_target_rate values
    (5/7 vs 6/7) so a leaked "always use the last-iterated mode" bug is unambiguously caught --
    the buggy version would have rendered BOTH as "6/7 (85.7%)" (panel's value), since `dict`
    iteration order in this call is insertion order and "panel" is inserted after "page".
    """
    page_report = _report(sample_count=7, usable_num=5, usable_den=7)
    panel_report = _report(sample_count=7, usable_num=6, usable_den=7)
    reports = {"page": page_report, "panel": panel_report}

    from dataclasses import asdict

    serialized = {"page": asdict(page_report), "panel": asdict(panel_report)}

    _render_rates_in_place(reports, serialized)

    assert serialized["page"]["usable_target_rate"]["rendered"] == str(Rate(5, 7))
    assert serialized["panel"]["usable_target_rate"]["rendered"] == str(Rate(6, 7))
    # Explicitly the failure mode a regression would reintroduce: page must NOT show panel's
    # rendered string.
    assert serialized["page"]["usable_target_rate"]["rendered"] != str(Rate(6, 7))


def test_render_rates_in_place_only_touches_rate_shaped_dicts():
    """Non-Rate fields (e.g. `analysis_mode`, `sample_count`) must pass through unmodified --

    the function's `isinstance(value, dict) and "numerator" in value and "denominator" in value`
    guard is what makes that distinction; this locks in that it doesn't over-match.
    """
    report = _report(sample_count=3, usable_num=1, usable_den=3)
    reports = {"page": report}
    from dataclasses import asdict

    serialized = {"page": asdict(report)}

    _render_rates_in_place(reports, serialized)

    assert serialized["page"]["analysis_mode"] == "page"
    assert serialized["page"]["sample_count"] == 3
    assert "rendered" not in str(serialized["page"]["analysis_mode"])
