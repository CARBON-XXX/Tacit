"""Broader decision training with protected validation tasks and resumable state.

The selected model must improve macro task NLL without exceeding the declared
regression allowance on original NLI and workflows. Test is evaluated only after
training and checkpoint selection. Checkpoint binaries remain research artifacts.
"""

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from general_data import load_corpus
from train_general import batch_loss, forward, prepare
from train_semantic import fitted_temperatures
from transformers import AutoTokenizer
from typed_common import record, summarize

from tacit.forked import ForkedTacit


def memory_batches(rows, batch_size, rng, max_tokens=16384, max_cells=16777216):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["length"] + 63) // 64].append(row)
    batches = []
    for bucket, values in buckets.items():
        maximum = bucket * 64
        if maximum > max_tokens or maximum**2 > max_cells:
            raise ValueError("one input exceeds the declared training memory bound")
        size = max(1, min(batch_size, max_tokens // maximum, max_cells // maximum**2))
        rng.shuffle(values)
        batches.extend(values[i : i + size] for i in range(0, len(values), size))
    rng.shuffle(batches)
    return batches


def track(row):
    if row["id"].startswith("mnli/"):
        layout = "structured" if row["id"].endswith("/structured") else "text"
        return "nli/" + layout + "/" + row["type"]
    for name in ["news", "emotion"]:
        if row["id"].startswith(name + "/"):
            return name
    return "typed"


def measurements(records):
    groups = defaultdict(list)
    for row in records:
        groups[track(row)].append(row)
    return {k: summarize(v) for k, v in groups.items()}


def selection(metrics, baseline, allowance):
    protected = ["nli/text/choice", "nli/text/noul", "typed"]
    violations = {
        k: metrics[k]["overall"]["soft_nll"] - baseline[k]["overall"]["soft_nll"]
        for k in protected
        if metrics[k]["overall"]["soft_nll"] > baseline[k]["overall"]["soft_nll"] + allowance
    }
    objective = sum(r["overall"]["soft_nll"] for r in metrics.values()) / len(metrics)
    return objective, violations


@torch.inference_mode()
def evaluate(model, rows, tokenizer, device, batch_size):
    model.eval()
    records, collected = [], {k: [] for k in ["noul", "choice", "score"]}
    for batch in memory_batches(rows, batch_size, random.Random(0)):
        logits = forward(model, batch, tokenizer, device).cpu()
        for row, z in zip(batch, logits, strict=True):
            for name, kind, _, start, end in row["meta"]:
                raw = z[start:end]
                records.append(record(row["case"], name, raw.softmax(-1).tolist()))
                collected[kind].append((raw.clone(), row["targets"][name]))
    return records, collected


def atomic_save(value, path):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path(".cache/general-v2"))
    parser.add_argument("--encoder", default=".cache/modernbert-base")
    parser.add_argument("--init-run", type=Path, default=Path("results/general/mixed-seed0"))
    parser.add_argument("--output", type=Path, default=Path("results/general/expanded-seed0"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workflow-repeat", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--encoder-lr", type=float, default=8e-6)
    parser.add_argument("--head-lr", type=float, default=8e-5)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--regression-allowance", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.workflow_repeat < 1 or args.validate_every < 1:
        raise ValueError("epochs, repeat and validation interval must be positive")
    if args.output.exists() and not args.resume:
        raise ValueError("use a fresh run directory or explicitly resume")
    args.output.mkdir(parents=True, exist_ok=True)
    arguments = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"
    }
    saved_manifest = (
        json.loads((args.output / "manifest.json").read_text()) if args.resume else None
    )
    if saved_manifest and saved_manifest["arguments"] != {**arguments, "architecture": "forked"}:
        raise ValueError("resume requires identical training arguments")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    parts = {k: load_corpus(args.corpus, k) for k in ["train", "validation", "calibration"]}
    print("preparing", {k: len(v) for k, v in parts.items()}, flush=True)
    train = prepare(parts["train"], tokenizer, args.max_length)
    replay = [r for r in train if not r["case"]["id"].startswith(("mnli/", "news/", "emotion/"))]
    train.extend(replay * (args.workflow_repeat - 1))
    validation = prepare(parts["validation"], tokenizer, args.max_length)
    calibration = prepare(parts["calibration"], tokenizer, args.max_length)
    model = ForkedTacit.from_encoder(args.encoder).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {"params": model.readout.parameters(), "lr": args.head_lr},
        ],
        weight_decay=0.01,
    )
    steps_per_epoch = len(memory_batches(train, args.batch_size, random.Random(0)))
    total_steps = args.epochs * steps_per_epoch
    warmup = min(200, total_steps // 10)

    def schedule(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        ratio = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * ratio))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    sources = [
        Path(__file__),
        Path("benchmarks/train_general.py"),
        Path("benchmarks/general_v2_data.py"),
        Path("src/tacit/forked.py"),
    ]
    source_sha = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest = {
        "arguments": {**arguments, "architecture": "forked"},
        "corpus_manifest": json.loads((args.corpus / "manifest.json").read_text()),
        "parameters": sum(p.numel() for p in model.parameters()),
        "hardware": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        "torch": torch.__version__,
        "source_sha256": source_sha,
        "training_cases_after_replay": len(train),
        "steps_per_epoch": steps_per_epoch,
        "init_checkpoint_sha256": hashlib.sha256(
            (args.init_run / "best.pt").read_bytes()
        ).hexdigest(),
        "selection": (
            "macro NLL over seven validation tracks, constrained original NLI/typed regression"
        ),
        "evaluation": "test after all checkpoint selection and held-out temperature fitting",
    }
    global_step, start_epoch, skip_batches, elapsed_before = 0, 1, 0, 0.0
    if args.resume:
        if (
            saved_manifest["source_sha256"] != source_sha
            or saved_manifest["corpus_manifest"] != manifest["corpus_manifest"]
        ):
            raise ValueError("source or corpus changed; cannot silently resume")
        checkpoint = torch.load(args.output / "resume.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step, start_epoch, skip_batches = (
            checkpoint["global_step"],
            checkpoint["epoch"],
            checkpoint["next_batch"],
        )
        rng.setstate(checkpoint["epoch_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state_all([r.cpu() for r in checkpoint["cuda_rng"]])
        baseline, history, best = checkpoint["baseline"], checkpoint["history"], checkpoint["best"]
        elapsed_before = checkpoint["elapsed_seconds"]
        print("resumed", global_step, start_epoch, skip_batches, flush=True)
    else:
        model.load_state_dict(
            torch.load(args.init_run / "best.pt", map_location=device, weights_only=True)[
                "state_dict"
            ]
        )
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        records, _ = evaluate(model, validation, tokenizer, device, args.batch_size)
        baseline = measurements(records)
        best, _ = selection(baseline, baseline, args.regression_allowance)
        history = [
            {"step": 0, "epoch": 0, "validation": baseline, "selection_nll": best, "eligible": True}
        ]
        atomic_save(
            {"state_dict": model.state_dict(), "step": 0, "epoch": 0}, args.output / "best.pt"
        )
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print("baseline", {k: v["overall"] for k, v in baseline.items()}, flush=True)
    start_time = time.perf_counter()
    print("training cases", len(train), "steps_per_epoch", steps_per_epoch, flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_rng_state = rng.getstate()
        epoch_batches = memory_batches(train, args.batch_size, rng)
        if len(epoch_batches) != steps_per_epoch:
            raise ValueError("batch count changed")
        model.train()
        for index, batch in enumerate(epoch_batches):
            if epoch == start_epoch and index < skip_batches:
                continue
            loss = batch_loss(forward(model, batch, tokenizer, device), batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            if global_step % 100 == 0:
                print(
                    "train",
                    epoch,
                    index + 1,
                    steps_per_epoch,
                    "step",
                    global_step,
                    "loss",
                    float(loss.detach()),
                    "elapsed",
                    elapsed_before + time.perf_counter() - start_time,
                    flush=True,
                )
            if global_step % args.validate_every == 0 or index + 1 == len(epoch_batches):
                records, _ = evaluate(model, validation, tokenizer, device, args.batch_size)
                metrics = measurements(records)
                objective, violations = selection(metrics, baseline, args.regression_allowance)
                item = {
                    "step": global_step,
                    "epoch": epoch,
                    "selection_nll": objective,
                    "eligible": not violations,
                    "violations": violations,
                    "validation": metrics,
                    "elapsed_seconds": elapsed_before + time.perf_counter() - start_time,
                }
                history.append(item)
                (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
                if not violations and objective < best:
                    best = objective
                    atomic_save(
                        {"state_dict": model.state_dict(), "step": global_step, "epoch": epoch},
                        args.output / "best.pt",
                    )
                atomic_save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "next_batch": index + 1,
                        "epoch_rng_state": epoch_rng_state,
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                        "baseline": baseline,
                        "history": history,
                        "best": best,
                        "elapsed_seconds": item["elapsed_seconds"],
                    },
                    args.output / "resume.pt",
                )
                print(
                    "validation",
                    global_step,
                    "objective",
                    objective,
                    "violations",
                    violations,
                    "accuracy",
                    {k: v["overall"]["accuracy"] for k, v in metrics.items()},
                    flush=True,
                )
                model.train()
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    _, collected = evaluate(model, calibration, tokenizer, device, args.batch_size)
    temperatures = fitted_temperatures(collected)
    test = prepare(load_corpus(args.corpus, "test"), tokenizer, args.max_length)
    records, _ = evaluate(model, test, tokenizer, device, args.batch_size)
    raw = measurements(records)
    for r in records:
        p = torch.tensor(r["probabilities"]).clamp_min(1e-12)
        r["probabilities"] = (p.log() / temperatures[r["type"]]).softmax(-1).tolist()
    result = {
        "selected_step": checkpoint["step"],
        "selected_epoch": checkpoint["epoch"],
        "temperatures": temperatures,
        "raw": raw,
        "calibrated": measurements(records),
        "records": records,
        "wall_seconds": elapsed_before + time.perf_counter() - start_time,
        "mode": "multi-task research model; task supervision and dataset license limits apply",
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("FINAL", {k: v["overall"] for k, v in result["calibrated"].items()}, flush=True)


if __name__ == "__main__":
    main()
