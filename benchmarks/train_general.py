"""Train instruction-conditioned NLI decisions with replay of known workflows."""

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from general_data import load_corpus
from train_semantic import fitted_temperatures
from transformers import AutoTokenizer
from typed_common import canonical_state, record, summarize, target

from tacit import decision_loss
from tacit.forked import ForkedTacit, pack_fork_batch
from tacit.semantics import schema_candidates


def prepare(cases, tokenizer, maximum):
    rows = []
    cache = {}
    for case in cases:
        key = json.dumps(case["questions"], ensure_ascii=False)
        if key not in cache:
            texts, meta = schema_candidates(case["questions"])
            tokens = tokenizer(texts, add_special_tokens=False, return_token_type_ids=False)[
                "input_ids"
            ]
            if any(len(t) > 254 for t in tokens):
                raise ValueError("candidate exceeds token limit")
            branches = [[tokenizer.mask_token_id, *t, tokenizer.sep_token_id] for t in tokens]
            cache[key] = (branches, meta)
        tokens = tokenizer(canonical_state(case["state"]), return_token_type_ids=False)["input_ids"]
        if len(tokens) > maximum:
            raise ValueError("state exceeds token limit")
        branches, meta = cache[key]
        rows.append(
            {
                "case": case,
                "tokens": tokens,
                "branches": branches,
                "meta": meta,
                "targets": {
                    name: torch.tensor(target(case, name), dtype=torch.float32) for name, *_ in meta
                },
                "length": len(tokens) + sum(map(len, branches)),
            }
        )
    return rows


def batches(rows, batch_size, rng):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["length"] // 128].append(row)
    result = []
    for values in buckets.values():
        rng.shuffle(values)
        result.extend(values[i : i + batch_size] for i in range(0, len(values), batch_size))
    rng.shuffle(result)
    return result


def forward(model, rows, tokenizer, device):
    packed = pack_fork_batch(
        [r["tokens"] for r in rows], [r["branches"] for r in rows], tokenizer.pad_token_id
    )
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        return model(**{k: v.to(device) for k, v in packed.items()})


def batch_loss(logits, rows):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        for name, kind, _, start, end in row["meta"]:
            groups[(kind, start, end)].append((i, row["targets"][name]))
    losses = []
    count = 0
    for (kind, start, end), members in groups.items():
        indices = [i for i, _ in members]
        gold = torch.stack([g for _, g in members]).to(logits.device)
        losses.append(
            len(indices)
            * decision_loss(
                logits[indices, start:end],
                gold,
                brier_weight=0.1,
                ordinal_weight=0.1 if kind == "score" else 0,
            )
        )
        count += len(indices)
    return torch.stack(losses).sum() / count


@torch.inference_mode()
def evaluate(model, rows, tokenizer, device, batch_size):
    model.eval()
    records = []
    collected = {k: [] for k in ["noul", "choice", "score"]}
    for batch in batches(rows, batch_size, random.Random(0)):
        logits = forward(model, batch, tokenizer, device).cpu()
        for row, z in zip(batch, logits, strict=True):
            for name, kind, _, start, end in row["meta"]:
                raw = z[start:end]
                records.append(record(row["case"], name, raw.softmax(-1).tolist()))
                collected[kind].append((raw.clone(), row["targets"][name]))
    return records, collected


def task_metrics(records):
    nli = [r for r in records if r["workflow"].startswith("mnli/")]
    typed = [r for r in records if not r["workflow"].startswith("mnli/")]
    metrics = {"nli": summarize(nli), "typed": summarize(typed)}
    metrics["selection_nll"] = sum(v["overall"]["soft_nll"] for v in metrics.values()) / 2
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path(".cache/general-v1"))
    parser.add_argument("--encoder", default=".cache/modernbert-base")
    parser.add_argument("--init-run", type=Path, default=Path("results/typed/forked-refine-seed0"))
    parser.add_argument("--output", type=Path, default=Path("results/general/mixed-seed0"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--replay-repeat", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise ValueError("use a fresh run directory")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    parts = {k: load_corpus(args.corpus, k) for k in ["train", "validation", "calibration"]}
    train = prepare(parts["train"], tokenizer, args.max_length)
    replay = [r for r in train if not r["case"]["workflow"].startswith("mnli/")]
    train.extend(replay * (args.replay_repeat - 1))
    validation = prepare(parts["validation"], tokenizer, args.max_length)
    calibration = prepare(parts["calibration"], tokenizer, args.max_length)
    model = ForkedTacit.from_encoder(args.encoder).to(device)
    model.load_state_dict(
        torch.load(args.init_run / "best.pt", map_location=device, weights_only=True)["state_dict"]
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {"params": model.readout.parameters(), "lr": args.head_lr},
        ],
        weight_decay=0.01,
    )
    steps = len(batches(train, args.batch_size, random.Random(0)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * steps)
    manifest = {
        "arguments": {
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "architecture": "forked",
        },
        "corpus_manifest": json.loads((args.corpus / "manifest.json").read_text()),
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__,
        "hardware": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        "training_cases_after_replay": len(train),
        "steps_per_epoch": steps,
        "init_checkpoint_sha256": hashlib.sha256(
            (args.init_run / "best.pt").read_bytes()
        ).hexdigest(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path("benchmarks/general_data.py"),
                Path("src/tacit/forked.py"),
            ]
        },
        "selection": "mean NLI/typed validation NLL; no test selection",
        "objective": "CE + .1 Brier + .1 score RPS; equal decision weights; typed replay",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    start_time = time.perf_counter()
    initial, _ = evaluate(model, validation, tokenizer, device, args.batch_size)
    metrics = task_metrics(initial)
    best = metrics["selection_nll"]
    selected = 0
    history = [
        {"epoch": 0, "validation": metrics, "elapsed_seconds": time.perf_counter() - start_time}
    ]
    torch.save({"state_dict": model.state_dict(), "epoch": 0}, args.output / "best.pt")
    print(
        "initial",
        json.dumps({k: v["overall"] for k, v in metrics.items() if k != "selection_nll"}),
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for step, batch in enumerate(batches(train, args.batch_size, rng)):
            logits = forward(model, batch, tokenizer, device)
            loss = batch_loss(logits, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * len(batch)
            count += len(batch)
            if step % 100 == 0:
                print("train", epoch, step, steps, "loss", float(loss.detach()), flush=True)
        records, _ = evaluate(model, validation, tokenizer, device, args.batch_size)
        metrics = task_metrics(records)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / count,
                "validation": metrics,
                "elapsed_seconds": time.perf_counter() - start_time,
            }
        )
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(
            "validation",
            epoch,
            json.dumps({k: v["overall"] for k, v in metrics.items() if k != "selection_nll"}),
            flush=True,
        )
        if metrics["selection_nll"] < best:
            best, selected = metrics["selection_nll"], epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, args.output / "best.pt")
    model.load_state_dict(
        torch.load(args.output / "best.pt", map_location=device, weights_only=True)["state_dict"]
    )
    _, collected = evaluate(model, calibration, tokenizer, device, args.batch_size)
    temperatures = fitted_temperatures(collected)
    test = prepare(load_corpus(args.corpus, "test"), tokenizer, args.max_length)
    records, _ = evaluate(model, test, tokenizer, device, args.batch_size)
    raw = task_metrics(records)
    for r in records:
        p = torch.tensor(r["probabilities"]).clamp_min(1e-12)
        r["probabilities"] = (p.log() / temperatures[r["type"]]).softmax(-1).tolist()
    result = {
        "selected_epoch": selected,
        "temperatures": temperatures,
        "raw": raw,
        "calibrated": task_metrics(records),
        "records": records,
        "wall_seconds": time.perf_counter() - start_time,
        "mode": (
            "NLI supervised training plus typed-workflow replay; new domains evaluated; "
            "not proven arbitrary-task generality"
        ),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "final",
        selected,
        json.dumps(
            {k: v["overall"] for k, v in result["calibrated"].items() if k != "selection_nll"}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
