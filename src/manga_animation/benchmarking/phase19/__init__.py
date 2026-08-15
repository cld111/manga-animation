"""Phase 19: OMG-LLaVA autonomous animation-target segmentation benchmark.

Evaluates whether the official OMG-LLaVA (referring-segmentation multimodal model) can replace
the current semantic grounding -> candidate selection -> segmentation chain for manga
animation-target discovery:

    FULL MANGA PAGE -> OMG-LLaVA -> animation target + pixel mask

Pure modules (prompts, parsing, masks, metrics, failure taxonomy, descriptions, model_meta)
are importable and testable on the local dev machine; the GPU adapter and the runners lazy-
import the heavy stack exactly like the rest of the project (ADR 0003).
"""
