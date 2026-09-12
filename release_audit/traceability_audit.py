#!/usr/bin/env python3
"""Audit the released Full table against 24 archived records and source logs.

Summary/record agreement is checked within 1e-6 percentage points. Historical
logs print accuracy to five decimals, so log/record agreement uses 1e-5 in
accuracy units. This does not audit unreleased ablations or pretrained results.
"""
import csv
import json
from pathlib import Path
import runpy
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    runpy.run_path(str(ROOT / "scripts/verify_reported_results.py"), run_name="__main__")
    records = json.loads((ROOT / "reported/full_24_runs.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / "reported/protocol.json").read_text(encoding="utf-8"))
    with (ROOT / "reported/full_summary.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(protocol["configs"])
    assert {r["dataset"] for r in rows} == set(protocol["configs"])
    deviations = []
    for row in rows:
        values = [r["test_accuracy"] * 100 for r in records if r["dataset"] == row["dataset"]]
        assert sorted(map(int, row["seeds"].split(";"))) == protocol["seeds"]
        for column, expected in (("accuracy_mean_percent", statistics.mean(values)), ("sample_std_percent", statistics.stdev(values))):
            delta = abs(float(row[column]) - expected)
            assert delta < 1e-6, (row["dataset"], column, delta)
            deviations.append(delta)
    print(f"PASS: Full summary means/stds agree with 24 records; maximum deviation {max(deviations):.3g} percentage points.")
    print("Scope: released Full results only; source-log tolerance is 1e-5 accuracy units, reflecting printed precision.")


if __name__ == "__main__":
    main()
