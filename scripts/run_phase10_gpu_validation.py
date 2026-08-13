"""Phase 10 real GPU E2E validation: re-run the real production pipeline against exactly the
three Phase 9 samples that showed a real mid-cycle visual defect
(`realworld_marika_love_meter`, `realworld_wind_breaker_finish`,
`realworld_villainess_ending_scuffle`; see `docs/phase9-results.md` section 7.1 and
`docs/decisions/0017-phase10-meshwarp-direction-default-and-panel-default.md`), on the exact
commit carrying Phase 10's fix, to check whether the fix visibly resolves them.

**Run this on the remote Kaggle/Jupyter GPU worker, never locally** -- see ADR 0003 and
CLAUDE.md's standing policy. Reuses `manga_animation.evaluation.harness.run_one_sample`, the
same real pipeline invocation `scripts/run_phase9_evaluation.py` uses, restricted to a small,
targeted sample/mode list instead of the full 10-sample x 2-mode dataset (a full re-run is not
needed to check whether a specific, already-identified defect is fixed, and would cost far more
real GPU time for no additional evidence).

`marika_love_meter` is run in BOTH modes: `page` (the mode the original defect occurred in --
Phase 10's forensics did not find a code-level fix for its root cause, so this checks whether it
is unchanged, not expected to be fixed) and `panel` (the new default -- checks whether panel
mode's own independent grounding attempt is still safely rejecting it, as it did in Phase 9,
rather than newly rendering a defect). `wind_breaker_finish`/`villainess_ending_scuffle` are run
in `panel` mode only (the mode the original defects occurred in, and Phase 10's actual fix
target).

Usage (on the GPU worker, after `git pull`):
    uv run python scripts/run_phase10_gpu_validation.py

Writes `outputs/experiments/phase10_gpu_validation_<timestamp>.json` and, for every run that
reaches rendering, the actual video under
`outputs/videos/phase10_evidence/<mode>/<sample-stem>/` (git-ignored generated artifacts, per
ADR 0002 -- never committed; downloaded locally afterward for visual inspection).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from manga_animation.core.config import load_config
from manga_animation.core.logging import setup_logging
from manga_animation.evaluation.dataset import REALWORLD_DATASET_PATH, load_eval_dataset
from manga_animation.evaluation.harness import (
    environment_metadata,
    panel_detection_evidence,
    run_one_sample,
)
from manga_animation.pipeline.orchestrator import build_default_clients

_TARGETS: list[tuple[str, Literal["page", "panel"]]] = [
    ("realworld_marika_love_meter", "page"),
    ("realworld_marika_love_meter", "panel"),
    ("realworld_wind_breaker_finish", "panel"),
    ("realworld_villainess_ending_scuffle", "panel"),
]


def main() -> None:
    setup_logging(debug=False)
    config = load_config("kaggle")
    device = config.resolve_device()
    clients = build_default_clients(config)

    samples_by_id = {s.sample_id: s for s in load_eval_dataset(REALWORLD_DATASET_PATH)}
    missing = [sid for sid, _ in _TARGETS if sid not in samples_by_id]
    if missing:
        raise SystemExit(f"sample_id(s) not found in the real-world dataset: {missing}")

    out_dir = Path("outputs/videos/phase10_evidence")
    experiments_dir = Path("outputs/experiments")
    experiments_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = experiments_dir / f"phase10_gpu_validation_{timestamp}.json"

    results = []
    for sample_id, mode in _TARGETS:
        sample = samples_by_id[sample_id]
        panel_count, panel_sources = panel_detection_evidence(Path(sample.image_path))
        print(f"=== running {sample_id} ({mode} mode) ===", flush=True)
        outcome = run_one_sample(
            sample,
            mode,
            config,
            clients,
            out_dir / mode / Path(sample.image_path).stem,
            panel_count,
            panel_sources,
        )
        record = json.loads(outcome.model_dump_json())
        results.append(record)
        seam = None
        if outcome.render_summary is not None:
            seam = outcome.render_summary.seam_artifact_suspected
        print(
            f"[{mode}] {sample_id}: status={outcome.status} "
            f"failing_stage={outcome.failing_stage or 'ok'} seam_artifact_suspected={seam}",
            flush=True,
        )
        experiments_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "environment": environment_metadata(device),
                    "config_env": "kaggle",
                    "model_variants": config.model_variants,
                    "git_commit_note": "run against the commit checked out on this worker",
                    "results": results,
                },
                indent=2,
                default=str,
            )
        )

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
