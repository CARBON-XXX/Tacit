"""Export compact, independently rescored research snapshots for source control."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from typed_common import load_cases, record, summarize


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_predictions(destination, name, records, cases):
    """Save predictions once; obtain canonical targets from the pinned dataset."""
    rebuilt = []
    with (destination / (name + ".predictions.jsonl")).open("w") as out:
        for r in records:
            entry = {k: r[k] for k in ["id", "question", "labels", "probabilities"]}
            out.write(json.dumps(entry, separators=(",", ":")) + "\n")
            reconstructed = record(cases[r["id"]], r["question"], r["probabilities"])
            if reconstructed["labels"] != r["labels"]:
                raise ValueError("label ordering changed")
            rebuilt.append(reconstructed)
    return summarize(rebuilt)


def main():
    source = Path("results/typed")
    destination = Path("results/typed-snapshot")
    destination.mkdir(exist_ok=True)
    cases = {c["id"]: c for c in load_cases(split="test")}
    for name in ["semantic-seed0", "forked-seed0", "structured-seed0", "structured-no-text-seed0"]:
        run = source / name
        result = json.loads((run / "result.json").read_text())
        records = result.pop("records")
        for r in records:
            p = torch.tensor(r["probabilities"]).clamp_min(1e-12)
            r["probabilities"] = (p.log() / result["temperatures"][r["type"]]).softmax(-1).tolist()
        recalculated = export_predictions(destination, name, records, cases)
        for key, value in result["calibrated"]["overall"].items():
            if value is not None and not np.isclose(
                value, recalculated["overall"][key], rtol=1e-6, atol=1e-7
            ):
                raise ValueError("export changed metric " + key)
        write(
            destination / (name + ".json"),
            {
                "run": name,
                "manifest": json.loads((run / "manifest.json").read_text()),
                "selected_epoch": result["selected_epoch"],
                "temperatures": result["temperatures"],
                "raw_metrics": result["raw"],
                "calibrated_metrics": recalculated,
                "history": json.loads((run / "history.json").read_text()),
                "checkpoint_sha256": sha256(run / "best.pt"),
                "checkpoint_bytes": (run / "best.pt").stat().st_size,
                "weight_license": "MIT"
                if name.startswith("structured")
                else "Apache-2.0 (ModernBERT-derived)",
            },
        )
    paired = json.loads((source / "paired-forked-laya.json").read_text())
    for name, records in paired.pop("records").items():
        metrics = export_predictions(destination, "paired-" + name, records, cases)
        if metrics != paired["metrics"][name]:
            raise ValueError("paired metrics changed on export")
    laya = json.loads((source / "laya-typed-test.json").read_text())
    laya.pop("records")
    paired["laya_original_run_provenance"] = laya
    write(destination / "paired.json", paired)
    for name in ["data-audit.json", "summary.json"]:
        write(destination / name, json.loads((source / name).read_text()))
    inventory = {
        p.name: sha256(p)
        for p in sorted(destination.iterdir())
        if p.is_file() and p.name != "checksums.json"
    }
    write(destination / "checksums.json", inventory)
    print("Exported and independently rescored", len(inventory), "snapshot files.")


if __name__ == "__main__":
    main()
