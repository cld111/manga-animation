"""Oracle-classification of the descriptions available for the phase-17 benchmark targets.

Phase brief section 7: inspect the project's existing target descriptions and classify every
one as PRODUCTION-AVAILABLE or GT-DERIVED / ORACLE. Only PRODUCTION-AVAILABLE descriptions may
feed the primary controlled result.

For every one of the 64 phase-17 targets the project's only existing description is the
production grounding prompt built by `_prompt_from_label` from the manifest's `semantic_label`
(`character_body` -> `"character body."`). It is category-level and instance-agnostic -- exactly
what production has today. The spatial / semantic+spatial conditions (B/C) are derived from the
GT bbox and are explicitly GT-DERIVED diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

from manga_animation.benchmarking.phase17.manifest import BenchmarkManifest
from manga_animation.benchmarking.phase19.prompts import (
    GT_DERIVED,
    PRODUCTION_AVAILABLE,
    ControlledPrompt,
    controlled_prompt,
    production_expression,
)


@dataclass(frozen=True, slots=True)
class DescriptionClass:
    """The classification of one target's available descriptions."""

    sample_id: str
    production_description: str  # exactly what the production pipeline can build today
    production_prompt: str  # the exact production grounding prompt ("character body.")
    condition_d: ControlledPrompt  # the primary controlled prompt built from it
    condition_a: ControlledPrompt  # category-only (diagnostic)
    provenance_d: str = PRODUCTION_AVAILABLE
    provenance_a: str = GT_DERIVED  # "character" is the category name, not a produced text

    def as_dict(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "production_description": self.production_description,
            "production_prompt": self.production_prompt,
            "condition_d": self.condition_d.prompt,
            "condition_a": self.condition_a.prompt,
            "provenance_d": self.provenance_d,
            "provenance_a": self.provenance_a,
        }


def classify_manifest_descriptions(
    manifest: BenchmarkManifest,
) -> list[DescriptionClass]:
    """Classify every manifest sample's available descriptions. No model involved.

    `sample.prompt` is the exact production grounding prompt already committed in the manifest;
    `production_expression` strips its trailing period to make it a referring expression.
    Conditions B and C are never PRODUCTION-AVAILABLE (they need the GT bbox) and are therefore
    excluded from this module's PRODUCTION-AVAILABLE table by construction.
    """
    out: list[DescriptionClass] = []
    for sample in manifest.samples:
        prod_expr = production_expression(sample.semantic_label)
        out.append(
            DescriptionClass(
                sample_id=sample.sample_id,
                production_description=prod_expr,
                production_prompt=sample.prompt,
                condition_d=controlled_prompt(
                    "D", semantic_label=sample.semantic_label
                ),
                condition_a=controlled_prompt("A", semantic_label=sample.semantic_label),
            )
        )
    return out


def production_available_only(
    classifications: list[DescriptionClass],
) -> list[DescriptionClass]:
    """The subset whose condition-D description is PRODUCTION-AVAILABLE (the allowed primary
    set). With the current manifest this is all 64 targets -- the filter exists so the invariant
    stays explicit in the report."""
    return [c for c in classifications if c.provenance_d == PRODUCTION_AVAILABLE]
