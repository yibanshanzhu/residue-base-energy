from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_config(name: str) -> dict:
    with (ROOT / "configs" / name).open() as handle:
        return yaml.safe_load(handle)


def _without_modality_flags(config: dict) -> dict:
    result = deepcopy(config)
    result["model"].pop("use_esm")
    result["model"].pop("use_geometry")
    return result


def test_modality_configs_differ_only_in_input_flags():
    full = _load_config("ets_family_v1.yaml")
    esm_only = _load_config("ets_family_esm_only_v1.yaml")
    structure_only = _load_config("ets_family_structure_only_v1.yaml")

    assert full["model"]["use_esm"] is True
    assert full["model"]["use_geometry"] is True
    assert esm_only["model"]["use_esm"] is True
    assert esm_only["model"]["use_geometry"] is False
    assert structure_only["model"]["use_esm"] is False
    assert structure_only["model"]["use_geometry"] is True
    assert _without_modality_flags(full) == _without_modality_flags(esm_only)
    assert _without_modality_flags(full) == _without_modality_flags(structure_only)


def test_broad_modality_configs_differ_only_in_input_flags():
    variants = {
        (True, True): _load_config("broad_full_v1.yaml"),
        (True, False): _load_config("broad_esm_only_v1.yaml"),
        (False, True): _load_config("broad_structure_only_v1.yaml"),
        (False, False): _load_config("broad_aa_only_v1.yaml"),
    }
    reference = variants[(True, True)]

    for flags, config in variants.items():
        assert (config["model"]["use_esm"], config["model"]["use_geometry"]) == flags
        assert _without_modality_flags(config) == _without_modality_flags(reference)
