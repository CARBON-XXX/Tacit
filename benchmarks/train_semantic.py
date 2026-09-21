"""Shared-state semantic decision pilot. Select on validation, evaluate test once.

Uses only 900 of the 1,200 public training cases; 148 select the checkpoint and
152 fit temperature. The teacher-label benchmark is a specialist training pilot,
not proof of arbitrary-schema generalization or superiority to a general Jev.
"""

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from typed_common import (
    DATA_REVISION,
    canonical_state,
    load_cases,
    partition,
    record,
    summarize,
    target,
)

from tacit import decision_loss
from tacit.forked import ForkedTacit, pack_forks
from tacit.semantics import SemanticTacit, schema_candidates


def prepare(cases, tokenizer, maximum):
    groups = {}
    rows = []
    for case in cases:
        key = json.dumps(case["questions"], ensure_ascii=False)
        if key not in groups:
            texts, meta = schema_candidates(case["questions"])
            tokens = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            raw = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=254)[
                "input_ids"
            ]
            branches = [[tokenizer.mask_token_id, *r, tokenizer.sep_token_id] for r in raw]
            groups[key] = {"tokens": tokens, "meta": meta, "branches": branches}
        tokens = tokenizer(
            canonical_state(case["state"]),
            truncation=True,
            max_length=maximum,
            return_token_type_ids=False,
        )
        rows.append({"case": case, "key": key, "tokens": tokens})
    return rows, groups


def batches(rows, batch_size, rng):
    groups = {}
    for row in rows:
        groups.setdefault(row["key"], []).append(row)
    out = []
    for group in groups.values():
        group = list(group)
        rng.shuffle(group)
        out.extend(group[i : i + batch_size] for i in range(0, len(group), batch_size))
    rng.shuffle(out)
    return out


def to_device(tokens, device):
    return {k: v.to(device) for k, v in tokens.items()}


def forward(model, tokenizer, rows, groups, device):
    key = rows[0]["key"]
    if any(r["key"] != key for r in rows):
        raise ValueError("batch must share a question schema")
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        if isinstance(model, ForkedTacit):
            packed = pack_forks(
                [r["tokens"]["input_ids"] for r in rows],
                groups[key]["branches"],
                tokenizer.pad_token_id,
            )
            logits = model(**to_device(packed, device))
        else:
            states = tokenizer.pad([r["tokens"] for r in rows], padding=True, return_tensors="pt")
            candidates = groups[key]["tokens"]
            logits = model(to_device(states, device), to_device(candidates, device), cache_key=key)
    return logits, groups[key]["meta"]


@torch.no_grad()
def evaluate(model, tokenizer, rows, groups, device, batch_size):
    model.eval()
    records, logits_by_type = [], {k: [] for k in ["noul", "choice", "score"]}
    for batch in batches(rows, batch_size, random.Random(0)):
        logits, meta = forward(model, tokenizer, batch, groups, device)
        logits = logits.cpu()
        for name, kind, _, start, end in meta:
            z = logits[:, start:end]
            for row, raw, p in zip(batch, z, z.softmax(-1), strict=True):
                records.append(record(row["case"], name, p.tolist()))
                logits_by_type[kind].append((raw.clone(), torch.tensor(target(row["case"], name))))
    return records, logits_by_type


def fitted_temperatures(collected):
    result = {}
    for kind, items in collected.items():
        if not items:
            result[kind] = 1.0
            continue
        options = torch.cat([torch.logspace(-1, 1, 101), torch.ones(1)])
        loss = []
        for t in options:
            loss.append(
                sum(float(-(g * (z / t).log_softmax(-1)).sum()) for z, g in items) / len(items)
            )
        result[kind] = float(options[int(np.argmin(loss))])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default=".cache/modernbert-base")
    parser.add_argument("--architecture", choices=["late", "forked"], default="late")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument(
        "--init-run", type=Path, help="Warm-start weights; optimizer and schedule restart"
    )
    parser.add_argument("--output", type=Path, default=Path("results/typed/semantic-seed0"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    parts = partition(load_cases())
    train, schemas = prepare(parts["train"], tokenizer, args.max_length)
    validation, _ = prepare(parts["validation"], tokenizer, args.max_length)
    calibration, _ = prepare(parts["calibration"], tokenizer, args.max_length)
    constructor = ForkedTacit if args.architecture == "forked" else SemanticTacit
    model = constructor.from_encoder(args.encoder).to(device)
    initialization = None
    if args.init_run:
        source = json.loads((args.init_run / "manifest.json").read_text())
        if source["arguments"].get("architecture", "late") != args.architecture:
            raise ValueError("warm start must use the same architecture")
        if source["arguments"]["max_length"] != args.max_length:
            raise ValueError("warm start must retain the input limit")
        source_checkpoint = args.init_run / "best.pt"
        model.load_state_dict(
            torch.load(source_checkpoint, map_location=device, weights_only=True)["state_dict"]
        )
        initialization = {
            "run": str(args.init_run),
            "checkpoint_sha256": hashlib.sha256(source_checkpoint.read_bytes()).hexdigest(),
            "optimizer": "fresh AdamW; fresh cosine schedule",
        }
    if args.freeze_encoder:
        model.encoder.requires_grad_(False)
    groups = [{"params": model.readout.parameters(), "lr": args.head_lr}]
    if not args.freeze_encoder:
        groups.append({"params": model.encoder.parameters(), "lr": args.encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    steps_per_epoch = len(batches(train, args.batch_size, random.Random(0)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * steps_per_epoch
    )
    manifest = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "dataset_revision": DATA_REVISION,
        "encoder_revision": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
        "split_ids": {k: [c["id"] for c in v] for k, v in parts.items()},
        "parameters": sum(p.numel() for p in model.parameters()),
        "hardware": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        "torch": torch.__version__,
        "objective": "soft CE + .1 Brier + .1 score RPS",
        "selection": "lowest validation soft NLL; test untouched until selection and calibration",
        "initialization": initialization,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                *sorted(Path("src/tacit").glob("*.py")),
                Path("benchmarks/typed_common.py"),
            ]
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    best, best_epoch, history = float("inf"), None, []
    start_time = time.perf_counter()
    if initialization:
        initial_records, _ = evaluate(
            model, tokenizer, validation, schemas, device, args.batch_size
        )
        initial_metrics = summarize(initial_records)
        best, best_epoch = initial_metrics["overall"]["soft_nll"], 0
        history.append({"epoch": 0, "validation": initial_metrics, "elapsed_seconds": 0.0})
        torch.save({"state_dict": model.state_dict(), "epoch": 0}, args.output / "best.pt")
    for epoch in range(args.epochs):
        model.train()
        if args.freeze_encoder:
            model.encoder.eval()
        total, count = 0.0, 0
        for step, batch in enumerate(batches(train, args.batch_size, rng)):
            logits, meta = forward(model, tokenizer, batch, schemas, device)
            losses = []
            for name, kind, _, start, end in meta:
                gold = torch.tensor(
                    np.stack([target(r["case"], name) for r in batch]), device=device
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
            scheduler.step()
            total += float(loss.detach()) * len(batch)
            count += len(batch)
            if step % 20 == 0:
                print(
                    "train",
                    epoch + 1,
                    step,
                    steps_per_epoch,
                    "loss",
                    float(loss.detach()),
                    flush=True,
                )
        records, _ = evaluate(model, tokenizer, validation, schemas, device, args.batch_size)
        metrics = summarize(records)
        nll = metrics["overall"]["soft_nll"]
        item = {
            "epoch": epoch + 1,
            "train_loss": total / count,
            "validation": metrics,
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        history.append(item)
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print("validation", json.dumps(item), flush=True)
        if nll < best:
            best, best_epoch = nll, epoch + 1
            torch.save(
                {"state_dict": model.state_dict(), "epoch": best_epoch}, args.output / "best.pt"
            )
    saved = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    _, calibration_logits = evaluate(
        model, tokenizer, calibration, schemas, device, args.batch_size
    )
    temperatures = fitted_temperatures(calibration_logits)
    # Test never selects a checkpoint within a run. Across research iterations it
    # remains an exploratory public benchmark, not a final untouched gate.
    test_cases = load_cases(split="test")
    test, test_schemas = prepare(test_cases, tokenizer, args.max_length)
    records, _ = evaluate(model, tokenizer, test, test_schemas, device, args.batch_size)
    calibrated = copy.deepcopy(records)
    for r in calibrated:
        p = torch.tensor(r["probabilities"]).clamp_min(1e-12)
        r["probabilities"] = (p.log() / temperatures[r["type"]]).softmax(-1).tolist()
    report = {
        "selected_epoch": best_epoch,
        "temperatures": temperatures,
        "raw": summarize(records),
        "calibrated": summarize(calibrated),
        "records": records,
        "wall_seconds": time.perf_counter() - start_time,
        "mode": "specialist",
    }
    model.eval()
    times = []
    with torch.no_grad():
        for row in test[:3]:
            forward(model, tokenizer, [row], test_schemas, device)
        for row in test:
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            forward(model, tokenizer, [row], test_schemas, device)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(1000 * (time.perf_counter() - start))
    report["latency"] = {
        "p50_ms": float(np.median(times)),
        "p95_ms": float(np.quantile(times, 0.95)),
        "scope": "pretokenized warm forward+padding/transfers, all five questions; not full SDK",
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("final", json.dumps({k: v for k, v in report.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
