"""Train a small observable-state specialist; fit everything on train only."""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from typed_common import DATA_REVISION, load_cases, partition, record, summarize, target

from tacit import decision_loss
from tacit.semantics import schema_candidates
from tacit.structured import StructuredTacit, StructuredVectorizer, schema_key


def prepare(cases, vectorizer):
    return {
        "cases": cases,
        "features": vectorizer.transform([c["state"] for c in cases]),
        "keys": [schema_key(c["questions"]) for c in cases],
    }


def batches(data, batch_size, rng):
    groups = {}
    for i, key in enumerate(data["keys"]):
        groups.setdefault(key, []).append(i)
    result = []
    for key, indices in groups.items():
        rng.shuffle(indices)
        result.extend(
            (key, indices[i : i + batch_size]) for i in range(0, len(indices), batch_size)
        )
    rng.shuffle(result)
    return result


@torch.no_grad()
def evaluate(model, data, metas, temperatures=None):
    model.eval()
    records = []
    for key, indices in batches(data, 64, random.Random(0)):
        logits = model(data["features"][indices], key)
        for name, kind, _, start, end in metas[key]:
            t = 1.0 if temperatures is None else temperatures[kind]
            probabilities = (logits[:, start:end] / t).softmax(-1)
            for i, p in zip(indices, probabilities, strict=True):
                records.append(record(data["cases"][i], name, p.tolist()))
    return records


def fit_temperatures(records):
    temperatures = {}
    for kind in ["choice", "noul", "score"]:
        subset = [r for r in records if r["type"] == kind]
        options = np.r_[np.logspace(-1, 1, 101), 1.0]
        losses = []
        for t in options:
            total = 0.0
            for r in subset:
                z = np.log(np.maximum(r["probabilities"], 1e-12)) / t
                logp = z - np.logaddexp.reduce(z)
                total -= float(np.dot(r["target"], logp))
            losses.append(total / max(1, len(subset)))
        temperatures[kind] = float(options[np.argmin(losses)]) if subset else 1.0
    return temperatures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--text-features", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=Path("results/typed/structured-seed0"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    rng = random.Random(args.seed)
    parts = partition(load_cases())
    vectorizer = StructuredVectorizer(args.text_features).fit([c["state"] for c in parts["train"]])
    data = {k: prepare(v, vectorizer) for k, v in parts.items()}
    metas = {
        schema_key(c["questions"]): schema_candidates(c["questions"])[1] for c in parts["train"]
    }
    model = StructuredTacit(vectorizer.width, {k: m[-1][-1] for k, m in metas.items()})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    manifest = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "dataset_revision": DATA_REVISION,
        "split_ids": {k: [c["id"] for c in v] for k, v in parts.items()},
        "parameters": sum(p.numel() for p in model.parameters()),
        "hardware": "CPU",
        "torch": torch.__version__,
        "feature_width": vectorizer.width,
        "objective": "soft CE + .1 Brier + .1 score RPS",
        "selection": (
            "lowest validation soft NLL; calibration separate; test exploratory after selection"
        ),
        "mode": "fixed-schema specialist, trained from scratch; observable state only",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "vectorizer.json").write_text(json.dumps(vectorizer.config, indent=2) + "\n")
    best, selected, history = float("inf"), 0, []
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for key, indices in batches(data["train"], 32, rng):
            logits = model(data["train"]["features"][indices], key)
            losses = []
            for name, kind, _, start, end in metas[key]:
                gold = torch.tensor(
                    np.stack([target(parts["train"][i], name) for i in indices]),
                    dtype=torch.float32,
                )
                losses.append(
                    decision_loss(
                        logits[:, start:end],
                        gold,
                        brier_weight=0.1,
                        ordinal_weight=0.1 if kind == "score" else 0,
                    )
                )
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation = summarize(evaluate(model, data["validation"], metas))
        nll = validation["overall"]["soft_nll"]
        history.append(
            {
                "epoch": epoch,
                "validation": validation,
                "elapsed_seconds": time.perf_counter() - start_time,
            }
        )
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if nll < best:
            best, selected = nll, epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, args.output / "best.pt")
        if epoch % 10 == 0:
            print(epoch, validation["overall"], flush=True)
        if epoch - selected >= args.patience:
            break
    model.load_state_dict(torch.load(args.output / "best.pt", weights_only=True)["state_dict"])
    temperatures = fit_temperatures(evaluate(model, data["calibration"], metas))
    test = prepare(load_cases(split="test"), vectorizer)
    records = evaluate(model, test, metas)
    calibrated = copy.deepcopy(records)
    for r in calibrated:
        p = torch.tensor(r["probabilities"]).clamp_min(1e-12)
        r["probabilities"] = (p.log() / temperatures[r["type"]]).softmax(-1).tolist()
    report = {
        "selected_epoch": selected,
        "temperatures": temperatures,
        "raw": summarize(records),
        "calibrated": summarize(calibrated),
        "records": records,
        "wall_seconds": time.perf_counter() - start_time,
        "mode": manifest["mode"],
    }
    # CPU timings are recorded separately by the paired evaluator when GPU training
    # is idle, avoiding machine contention as a source of claimed speed gains.
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("final", json.dumps({k: v for k, v in report.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
