"""Phase 19 OMG-LLaVA benchmark: smoke, five-target, full controlled, autonomous.

    FULL MANGA PAGE -> OMG-LLaVA -> target + pixel mask

Run on the Kaggle/Jupyter GPU worker (OMG-LLaVA is remote-GPU work, never local), after the
official OMG-LLaVA stack is installed and the weights are on disk (see
`scripts/setup_phase19_omg_llava_worker.py`). The model is loaded once per subcommand and
released deterministically.

Subcommands:

    smoke      one inference on one page (VRAM + latency + [SEG] emission check)
    five       the five difficult-target smoke test (phase brief section 16)
    full       the full 64-target controlled benchmark (sections 8/9/17)
    autonomous the autonomous-discovery run over a page subset (section 18)
    report     rebuild the report from saved per-sample results (no GPU)

Usage:

    python scripts/run_phase19_omg_llava_benchmark.py \
        --omg-config <omg_llava finetune config .py> \
        --omg-pth    /kaggle/working/models/omg_llava/omg_llava_7b_finetune_8gpus.pth \
        --manifest   configs/phase17_benchmark.yaml \
        --out        outputs/experiments/phase19_omg_llava \
        full --condition D

`--precision` maps to the LLM build strategy (official config default is 4-bit bitsandbytes;
`fp16` needs `--shard-two-gpus` on T4s). `--resolution` patches the config's image processor
(1024 is the official default; anything else is experimental).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from manga_animation.benchmarking.phase17.manifest import BenchmarkManifest, load_manifest
from manga_animation.benchmarking.phase19.adapter import OMGLLavaAdapter
from manga_animation.benchmarking.phase19.autonomous import (
    build_autonomous_gallery,
    run_autonomous_pages,
    save_autonomous_records,
)
from manga_animation.benchmarking.phase19.prompts import (
    CONTROLLED_CONDITIONS,
    autonomous_prompt,
    condition_provenance,
    controlled_prompt,
)
from manga_animation.benchmarking.phase19.report import (
    apply_forbidden,
    build_report,
    compute_forbidden_overlap,
    write_report,
)
from manga_animation.benchmarking.phase19.run import (
    load_records,
    run_controlled_benchmark,
    run_smoke,
    save_records,
    select_five_targets,
)
from manga_animation.evaluation.harness import environment_metadata

DEFAULT_MANIFEST = "configs/phase17_benchmark.yaml"
DEFAULT_OUT = "outputs/experiments/phase19_omg_llava"


def _trim_manifest(manifest: BenchmarkManifest, limit: int) -> BenchmarkManifest:
    return BenchmarkManifest(
        version=manifest.version,
        seed=manifest.seed,
        main_category=manifest.main_category,
        samples=manifest.samples[:limit],
    )


def _build_adapter(args, device: str) -> OMGLLavaAdapter:
    return OMGLLavaAdapter(
        config_path=args.omg_config,
        pth_path=args.omg_pth,
        device=device,
        llm_bits=args.precision,
        shard_two_gpus=args.shard_two_gpus,
        max_new_tokens=args.max_new_tokens,
        temperature=0.1,
        offload_folder=args.offload_folder,
    )


def _condition_prompt_for(manifest: BenchmarkManifest, condition: str):
    def prompt_for(sample) -> str:
        return controlled_prompt(
            condition,
            semantic_label=sample.semantic_label,
            gt_bbox=sample.gt_bbox,
            page_size=sample.page_size,
        ).prompt

    return prompt_for


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 19 OMG-LLaVA benchmark")
    parser.add_argument(
        "--omg-config", required=True, help="official omg_llava finetune config (.py)"
    )
    parser.add_argument("--omg-pth", required=True, help="omg_llava_7b_finetune_8gpus.pth path")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--precision", default="4", choices=["4", "8", "fp16"],
        help="LLM build strategy (official config default is 4-bit bitsandbytes)",
    )
    parser.add_argument("--shard-two-gpus", action="store_true",
                        help="device_map=auto for the LLM across the two T4s (fp16)")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="image processor size (1024 = official default)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--offload-folder", default=None)
    parser.add_argument("--limit", type=int, default=None, help="limit targets/pages")
    sub = parser.add_subparsers(dest="command", required=True)

    p_smoke = sub.add_parser("smoke", help="single inference smoke test")
    p_smoke.add_argument("--sample", default=None, help="sample_id to use (default first)")
    p_smoke.add_argument("--mode", choices=["controlled", "autonomous"], default="autonomous")

    sub.add_parser("five", help="five-target smoke test")

    p_full = sub.add_parser("full", help="full 64-target controlled benchmark")
    p_full.add_argument("--condition", choices=list(CONTROLLED_CONDITIONS), default="D")

    p_auto = sub.add_parser("autonomous", help="autonomous discovery over page subset")
    p_auto.add_argument("--limit-pages", type=int, default=None)

    sub.add_parser("report", help="rebuild the report from saved per-sample results")

    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    if args.limit is not None and args.command in ("full", "five"):
        manifest = _trim_manifest(manifest, args.limit)

    out_dir = Path(args.out)
    run_dir = out_dir / f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    dataset_dir = out_dir.parent / "phase17_object_segmentation" / "dataset"
    if not (dataset_dir / f"{manifest.samples[0].sample_id}.png").exists():
        dataset_dir = out_dir / "dataset"

    if args.command == "report":
        source = out_dir / "per_sample_results.json"
        if not source.exists():
            source = sorted(out_dir.glob("run_*/per_sample_results.json"))[-1]
        run_dir = source.parent
        records = load_records(run_dir)
        condition = records[0].condition if records else "?"
        provenance = records[0].provenance if records else "?"
        _finish_report(records, manifest, run_dir, out_dir, condition, provenance)
        return

    from manga_animation.core.config import load_config

    config = load_config()
    device = config.resolve_device()
    if device != "cuda":
        print(f"WARNING: resolved device is {device!r} -- expected cuda on the GPU worker")

    adapter = _build_adapter(args, device)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "command": args.command,
        "condition": getattr(args, "condition", None),
        "precision": args.precision,
        "shard_two_gpus": args.shard_two_gpus,
        "resolution": args.resolution,
        "max_new_tokens": args.max_new_tokens,
        "created_at": datetime.now(UTC).isoformat(),
        "environment": environment_metadata(device),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    if args.command == "smoke":
        sample = manifest.samples[0]
        if args.sample is not None:
            sample = next(s for s in manifest.samples if s.sample_id == args.sample)
        image = np.asarray(Image.open(dataset_dir / f"{sample.sample_id}.png").convert("RGB"))
        prompt = (
            autonomous_prompt()
            if args.mode == "autonomous"
            else controlled_prompt("D", semantic_label=sample.semantic_label).prompt
        )
        print(f"loading OMG-LLaVA (precision={args.precision}, shard={args.shard_two_gpus})")
        adapter.load()
        result = run_smoke(image, prompt, adapter, run_dir, label=sample.sample_id)
        adapter.unload()
        (run_dir / "smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))

    elif args.command == "five":
        five = select_five_targets(manifest)
        print("five smoke targets (difficulty proxies):")
        for s in five:
            print(f"  {s.sample_id}  area={s.features.get('area_fraction', 0):.4f} "
                  f"density={s.features.get('silhouette_density', 0):.3f} "
                  f"aspect={s.features.get('aspect_ratio', 0):.2f}")
        adapter.load()
        results = {}
        for sample in five:
            image = np.asarray(Image.open(dataset_dir / f"{sample.sample_id}.png").convert("RGB"))
            prompt = controlled_prompt("D", semantic_label=sample.semantic_label).prompt
            results[sample.sample_id] = run_smoke(
                image, prompt, adapter, run_dir, label=f"five_{sample.sample_id}"
            )
            results[sample.sample_id]["prompt"] = prompt
        adapter.unload()
        (run_dir / "five_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))

    elif args.command == "full":
        condition = args.condition
        provenance = condition_provenance(condition)
        print(f"full controlled benchmark, condition {condition} "
              f"({provenance}), {len(manifest.samples)} targets")
        adapter.load()
        try:
            records = run_controlled_benchmark(
                manifest,
                dataset_dir,
                run_dir,
                adapter,
                condition=condition,
                provenance=provenance,
                prompt_for=_condition_prompt_for(manifest, condition),
            )
        finally:
            adapter.unload()
        save_records(records, run_dir)
        _finish_report(records, manifest, run_dir, out_dir, condition, provenance)

    elif args.command == "autonomous":
        instruction = autonomous_prompt()
        n_pages = len({f"{s.book}_{s.page_index:03d}" for s in manifest.samples})
        print(f"autonomous discovery over {n_pages} pages")
        adapter.load()
        try:
            records = run_autonomous_pages(
                manifest,
                dataset_dir,
                run_dir,
                adapter,
                instruction=instruction,
                limit=args.limit_pages,
            )
        finally:
            adapter.unload()
        save_autonomous_records(records, run_dir)
        gallery = build_autonomous_gallery(records, run_dir)
        print(f"autonomous gallery: {len(gallery)} montages under {run_dir}")
        print(f"autonomous records: {run_dir / 'autonomous_pages.json'}")


def _finish_report(
    records,
    manifest,
    run_dir: Path,
    out_dir: Path,
    condition: str,
    provenance: str,
) -> None:
    """Shared report step: forbidden overlap (needs HF_TOKEN), taxonomy, report files."""
    hf_token = os.environ.get("HF_TOKEN")
    forbidden = {}
    if hf_token:
        forbidden = compute_forbidden_overlap(records, manifest, out_dir / "cache", hf_token)
        apply_forbidden(records, forbidden)
    report = build_report(records, condition=condition, provenance=provenance)
    json_path, md_path = write_report(report, records, run_dir)
    print(f"report: {json_path}\n        {md_path}")
    if forbidden:
        (run_dir / "forbidden_overlap.json").write_text(
            json.dumps(forbidden, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
