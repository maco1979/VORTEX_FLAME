#!/usr/bin/env python3
"""
Align 14 Souls to 10 JEPA Modalities
======================================

Adds JEPA configuration to each soul's YAML config:
  - jepa_modality: which JEPA this soul primarily uses
  - jepa_slots: slot names for this soul's domain
  - jepa_input_dim: expected input dimension
  - jepa_class: CAJEPA/CFINJEPA/CCODEJEPA etc.

Mapping:
  beethoven    → AUDIO    (CAJEPA)
  vangogh      → VISUAL   (CVJEPA)
  monet        → ART      (CARTJEPA)
  cezanne      → CODE     (CCODEJEPA)
  einstein     → PHYSICS  (CPHYSJEPA)
  galileo      → PHYSICS  (CPHYSJEPA)
  davinci      → DESIGN   (CDESIGNJEPA)
  darwin       → BIOLOGY  (CBIOJEPA)
  yuanlongping → BIOLOGY  (CBIOJEPA)
  humboldt     → GEOGRAPHY (CGEOJEPA)
  montesquieu  → LAW      (CLAWJEPA)
  strategy     → FINANCIAL (CFINJEPA)
  guizhu       → AUDIO+LAW (multi-modality)
  herodotus    → GEOGRAPHY+LAW (multi-modality)
"""

import os
import sys

SOUL_JEPA_MAP = {
    "beethoven": {
        "jepa_modality": "audio",
        "jepa_class": "CAJEPA",
        "jepa_input_dim": 512,
        "jepa_num_slots": 5,
        "jepa_slot_names": [
            "drums_percussion", "bass", "vocals", "melody_lead", "harmony_pads",
            "effects_ambient", "silence_gap",
        ],
        "jepa_secondary": [],
    },
    "vangogh": {
        "jepa_modality": "visual",
        "jepa_class": "CVJEPA",
        "jepa_input_dim": 768,
        "jepa_num_slots": 7,
        "jepa_slot_names": [
            "foreground_subject", "background_scene", "text_overlay",
            "color_palette", "composition", "lighting", "depth_layer",
        ],
        "jepa_secondary": ["art"],
    },
    "monet": {
        "jepa_modality": "art",
        "jepa_class": "CARTJEPA",
        "jepa_input_dim": 768,
        "jepa_num_slots": 8,
        "jepa_slot_names": [
            "composition", "color_harmony", "texture", "style_period",
            "emotional_tone", "symbolism", "technique", "light_quality",
        ],
        "jepa_secondary": ["visual"],
    },
    "cezanne": {
        "jepa_modality": "code",
        "jepa_class": "CCODEJEPA",
        "jepa_input_dim": 384,
        "jepa_num_slots": 7,
        "jepa_slot_names": [
            "control_flow", "data_structures", "api_calls",
            "error_handling", "side_effects", "type_system", "concurrency",
        ],
        "jepa_secondary": [],
    },
    "einstein": {
        "jepa_modality": "physics",
        "jepa_class": "CPHYSJEPA",
        "jepa_input_dim": 512,
        "jepa_num_slots": 7,
        "jepa_slot_names": [
            "kinematics", "forces", "energy", "fields",
            "conservation_laws", "boundary_conditions", "symmetry",
        ],
        "jepa_secondary": ["financial"],
    },
    "galileo": {
        "jepa_modality": "physics",
        "jepa_class": "CPHYSJEPA",
        "jepa_input_dim": 512,
        "jepa_num_slots": 7,
        "jepa_slot_names": [
            "kinematics", "forces", "energy", "fields",
            "conservation_laws", "boundary_conditions", "symmetry",
        ],
        "jepa_secondary": [],
    },
    "davinci": {
        "jepa_modality": "design",
        "jepa_class": "CDESIGNJEPA",
        "jepa_input_dim": 512,
        "jepa_num_slots": 6,
        "jepa_slot_names": [
            "layout_grid", "typography", "spacing_rhythm",
            "visual_hierarchy", "interaction_pattern", "accessibility",
        ],
        "jepa_secondary": ["visual"],
    },
    "darwin": {
        "jepa_modality": "biology",
        "jepa_class": "CBIOJEPA",
        "jepa_input_dim": 512,
        "jepa_num_slots": 6,
        "jepa_slot_names": [
            "gene_expression", "protein_structure", "metabolic_pathway",
            "cell_signal", "phenotype_trait", "evolutionary_pressure",
        ],
        "jepa_secondary": [],
    },
    "yuanlongping": {
        "jepa_modality": "biology",
        "jepa_class": "CBIOJEPA",
        "jepa_input_dim": 512,
        "jepa_num_slots": 6,
        "jepa_slot_names": [
            "gene_expression", "protein_structure", "metabolic_pathway",
            "cell_signal", "phenotype_trait", "evolutionary_pressure",
        ],
        "jepa_secondary": ["geography"],
    },
    "humboldt": {
        "jepa_modality": "geography",
        "jepa_class": "CGEOJEPA",
        "jepa_input_dim": 384,
        "jepa_num_slots": 6,
        "jepa_slot_names": [
            "terrain_elevation", "vegetation_cover", "water_systems",
            "climate_pattern", "human_activity", "geological_structure",
        ],
        "jepa_secondary": [],
    },
    "montesquieu": {
        "jepa_modality": "law",
        "jepa_class": "CLAWJEPA",
        "jepa_input_dim": 256,
        "jepa_num_slots": 6,
        "jepa_slot_names": [
            "legal_rule", "precedent_case", "jurisdiction_scope",
            "temporal_validity", "exception_clause", "interpretation_method",
        ],
        "jepa_secondary": [],
    },
    "strategy": {
        "jepa_modality": "financial",
        "jepa_class": "CFINJEPA",
        "jepa_input_dim": 256,
        "jepa_num_slots": 6,
        "jepa_slot_names": [
            "trend_direction", "momentum", "volatility",
            "volume_profile", "support_resistance", "market_regime",
        ],
        "jepa_secondary": [],
    },
    "guizhu": {
        "jepa_modality": "audio",
        "jepa_class": "CAJEPA",
        "jepa_input_dim": 512,
        "jepa_num_slots": 5,
        "jepa_slot_names": [
            "drums_percussion", "bass", "vocals", "melody_lead", "harmony_pads",
            "effects_ambient", "silence_gap",
        ],
        "jepa_secondary": ["law"],
    },
    "herodotus": {
        "jepa_modality": "geography",
        "jepa_class": "CGEOJEPA",
        "jepa_input_dim": 384,
        "jepa_num_slots": 6,
        "jepa_slot_names": [
            "terrain_elevation", "vegetation_cover", "water_systems",
            "climate_pattern", "human_activity", "geological_structure",
        ],
        "jepa_secondary": ["law"],
    },
}


def update_soul_config(soul_name: str, jepa_config: dict):
    config_dir = r"D:\VORTEX_FLAME\soul_config"
    config_path = os.path.join(config_dir, f"{soul_name}.yaml")

    if not os.path.exists(config_path):
        print(f"  SKIP {soul_name}: no config file at {config_path}")
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "jepa_modality" in content:
        print(f"  SKIP {soul_name}: already has JEPA config")
        return False

    jepa_section = "\n# JEPA Perception Layer\n"
    jepa_section += f"jepa_modality: \"{jepa_config['jepa_modality']}\"\n"
    jepa_section += f"jepa_class: \"{jepa_config['jepa_class']}\"\n"
    jepa_section += f"jepa_input_dim: {jepa_config['jepa_input_dim']}\n"
    jepa_section += f"jepa_num_slots: {jepa_config['jepa_num_slots']}\n"

    jepa_section += "jepa_slot_names:\n"
    for name in jepa_config["jepa_slot_names"]:
        jepa_section += f"  - \"{name}\"\n"

    if jepa_config["jepa_secondary"]:
        jepa_section += "jepa_secondary:\n"
        for sec in jepa_config["jepa_secondary"]:
            jepa_section += f"  - \"{sec}\"\n"

    with open(config_path, "a", encoding="utf-8") as f:
        f.write(jepa_section)

    print(f"  OK {soul_name}: +{jepa_config['jepa_class']} ({jepa_config['jepa_modality']})")
    return True


def main():
    print("=" * 60)
    print("14 Souls → 10 JEPA Modalities Alignment")
    print("=" * 60)

    updated = 0
    skipped = 0

    for soul_name, jepa_config in SOUL_JEPA_MAP.items():
        if update_soul_config(soul_name, jepa_config):
            updated += 1
        else:
            skipped += 1

    print(f"\nDone! {updated} updated, {skipped} skipped")

    print("\nAlignment Summary:")
    modality_count = {}
    for soul_name, jepa_config in SOUL_JEPA_MAP.items():
        mod = jepa_config["jepa_modality"]
        if mod not in modality_count:
            modality_count[mod] = []
        modality_count[mod].append(soul_name)

    for mod, souls in sorted(modality_count.items()):
        print(f"  {mod:12s}: {', '.join(souls)}")


if __name__ == "__main__":
    main()
