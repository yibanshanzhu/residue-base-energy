from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rbe.eval.summary import numeric_keys


def write_rows_tsv(path: str | Path, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = ["sample", "target_path", "pred_path"] + numeric_keys(rows)
    with output.open("w") as handle:
        handle.write("\t".join(keys) + "\n")
        for row in rows:
            handle.write("\t".join(_format_row_value(row.get(key, "")) for key in keys))
            handle.write("\n")


def write_summary_tsv(path: str | Path, summary: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        handle.write("metric\tmean\tstd\tn\n")
        for row in summary:
            handle.write(
                f"{row['metric']}\t{row['mean']:.6f}\t{row['std']:.6f}\t{row['n']}\n"
            )


def write_summary_json(
    path: str | Path, samples: list[dict], summary: list[dict]
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"samples": samples, "summary": summary}, indent=2),
        encoding="utf-8",
    )


def _format_row_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)
