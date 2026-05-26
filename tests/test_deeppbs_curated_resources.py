from __future__ import annotations

from pathlib import Path

from rbe.data.pwm import read_pwm
from scripts.prepare_deeppbs_smoke import (
    DEFAULT_CURATED_ROOT,
    curated_pwm_path,
    parse_deeppbs_entry,
    resolve_fold_file,
)


def test_parse_deeppbs_entry():
    assert parse_deeppbs_entry("6od5_A_ITF2_HUMAN.H11MO.0.C.npz") == (
        "6od5",
        "A",
        "ITF2_HUMAN.H11MO.0.C",
    )


def test_vendored_deeppbs_fold_and_pwm_exist():
    fold_file = resolve_fold_file("valid0.txt", DEFAULT_CURATED_ROOT)
    assert fold_file.name == "valid0.txt"
    assert fold_file.exists()

    pwm_path = curated_pwm_path(DEFAULT_CURATED_ROOT, "ITF2_HUMAN.H11MO.0.C")
    assert pwm_path.exists()
    assert read_pwm(pwm_path).shape[1] == 4
    assert str(DEFAULT_CURATED_ROOT.relative_to(Path.cwd())).startswith("resources")
