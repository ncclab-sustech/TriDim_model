#!/usr/bin/env python3
"""Check the published Full release without EEG data or training.

Requires PyYAML. Checks the eight-dataset registry, configs, retained split
indices, archived results and release hashes. Does not verify external EEG
stores or claim that every dataset has a retained split manifest.
"""
import json
from pathlib import Path
import runpy

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    protocol = json.loads((ROOT / "reported/protocol.json").read_text(encoding="utf-8"))
    datasets = {"AD65", "BCIC2A", "FACED_new", "Physionet_MI", "SEED", "SEED_V", "SHU", "SleepEDF_full"}
    assert set(protocol["configs"]) == datasets, "Unexpected dataset registry"
    assert protocol["model"] == "tridim" and protocol["seeds"] == [5, 42, 43]
    registered = {ROOT / rel for rel in protocol["configs"].values()}
    assert registered == set((ROOT / "configs/paper/full").glob("*.yaml")), "Unregistered or missing configuration"
    records = json.loads((ROOT / "reported/full_24_runs.json").read_text(encoding="utf-8"))
    for record in records:
        assert record["config"] == protocol["configs"][record["dataset"]], "Result/config mismatch"
    manifests = set()
    for dataset, rel in protocol["configs"].items():
        cfg = yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))
        params = cfg["params"]
        assert params["model"] == protocol["model"], dataset
        assert params["seeds"] == protocol["seeds"], dataset
        assert params["train_ratio"] == .8 and params["val_ratio"] == .1, dataset
        assert params["select_metric"] == "Accuracy", dataset
        for mapping in (cfg, params):
            for key in ("electrode_csv", "canonical_channel_coord_path", "input_channel_coord_path", "external_split_manifest"):
                if not mapping.get(key):
                    continue
                for seed in protocol["seeds"]:
                    path = (ROOT / mapping[key].format(seed=seed)).resolve()
                    assert path.is_relative_to(ROOT) and path.is_file(), (dataset, key)
                    if key != "external_split_manifest":
                        continue
                    manifest = json.loads(path.read_text(encoding="utf-8"))
                    assert manifest["seed"] == seed, path
                    splits = manifest["splits"]
                    indices = []
                    for split in ("train", "val", "test"):
                        values = splits[split]
                        assert values and all(type(i) is int and i >= 0 for i in values), (path, split)
                        assert len(values) == len(set(values)), (path, split, "duplicate index")
                        indices.append(set(values))
                    assert not any(indices[i] & indices[j] for i in range(3) for j in range(i)), (path, "overlapping splits")
                    manifests.add(path)
    runpy.run_path(str(ROOT / "scripts/verify_reported_results.py"), run_name="__main__")
    print(f"PASS: 8 dataset configurations and {len(manifests)} retained split manifests; archived results and release hashes verified.")


if __name__ == "__main__":
    main()
