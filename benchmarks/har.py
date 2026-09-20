"""Subject-disjoint UCI HAR pilot, raw 9-channel windows, no text or pretrained LM.

Run: PYTHONPATH=src python benchmarks/har.py --seeds 0 1 2
Downloads the public CC BY 4.0 dataset to .cache/har; does not redistribute it.
Hyperparameters and subject partitions are fixed before test scoring. Model
selection uses validation subjects only. Calibration subjects are separate.
"""

import argparse
import copy
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tacit import (
    Choice,
    ConformalPolicy,
    EncoderConfig,
    SignalConfig,
    SignalTacit,
    TemperatureScaler,
    decision_loss,
    expected_calibration_error,
)

URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip"
)
LABELS = ["walking", "walking_upstairs", "walking_downstairs", "sitting", "standing", "laying"]
CHANNELS = [
    f"{sensor}_{axis}" for sensor in ["body_acc", "body_gyro", "total_acc"] for axis in "xyz"
]


def dataset(cache):
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "uci-har.zip"
    if not archive.exists():
        print("Downloading UCI HAR (58 MB)", flush=True)
        temporary = archive.with_suffix(".part")
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        temporary.replace(archive)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    tensors = cache / "raw-windows.npz"
    if not tensors.exists():
        outer = zipfile.ZipFile(archive)
        if "UCI HAR Dataset.zip" in outer.namelist():
            inner = zipfile.ZipFile(io.BytesIO(outer.read("UCI HAR Dataset.zip")))
        else:
            inner = outer
        arrays = {}
        for split in ["train", "test"]:
            prefix = f"UCI HAR Dataset/{split}/"
            arrays[f"{split}_x"] = np.stack(
                [
                    np.loadtxt(
                        io.BytesIO(inner.read(f"{prefix}Inertial Signals/{c}_{split}.txt")),
                        dtype=np.float32,
                    )
                    for c in CHANNELS
                ],
                axis=-1,
            )
            for field, file in [("y", "y"), ("subject", "subject")]:
                arrays[f"{split}_{field}"] = np.loadtxt(
                    io.BytesIO(inner.read(f"{prefix}{file}_{split}.txt")), dtype=np.int64
                )
        np.savez_compressed(tensors, **arrays)
    with np.load(tensors) as saved:
        data = {k: torch.from_numpy(saved[k].copy()) for k in saved.files}
    data["train_y"] -= 1
    data["test_y"] -= 1
    subjects = sorted(data["train_subject"].unique().tolist())
    groups = {
        "train": subjects[:-6],
        "validation": subjects[-6:-4],
        "temperature": subjects[-4:-2],
        "policy": subjects[-2:],
    }
    assert not set(subjects) & set(data["test_subject"].tolist())
    splits = {}
    for name, ids in groups.items():
        mask = torch.isin(data["train_subject"], torch.tensor(ids))
        splits[name] = (data["train_x"][mask], data["train_y"][mask])
    mean = splits["train"][0].mean((0, 1))
    std = splits["train"][0].std((0, 1)).clamp_min(1e-5)
    splits["test"] = data["test_x"], data["test_y"]
    splits = {k: ((x - mean) / std, y) for k, (x, y) in splits.items()}
    manifest = {
        "url": URL,
        "sha256": checksum,
        "license": "CC BY 4.0",
        "doi": "10.24432/C54S4K",
        "subjects": groups,
        "test_subjects": sorted(data["test_subject"].unique().tolist()),
        "split_sizes": {k: len(x) for k, (x, _) in splits.items()},
        "channels": CHANNELS,
        "labels": LABELS,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "protocol": "128 preprocessed raw-signal samples per window; last-event decision; "
        "reset state between windows; no text; no pretrained weights; "
        "training-only normalization; subject-disjoint partitions",
    }
    return splits, manifest


class WindowBaseline(nn.Module):
    """Small trained reference with fixed mean/std/min/max window features."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(36, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 6)
        )

    def forward(self, x):
        return self.net(torch.cat([x.mean(1), x.std(1), x.amin(1), x.amax(1)], -1))


class TemporalBaseline(nn.Module):
    """Conventional learned temporal features, with the same raw input window."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(9, 64, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 6),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2))


def forward(model, x):
    return model(x)[0]["activity"][:, -1] if isinstance(model, SignalTacit) else model(x)


@torch.no_grad()
def collect(model, x, batch=128):
    model.eval()
    return torch.cat([forward(model, x[i : i + batch]).cpu() for i in range(0, len(x), batch)])


def metrics(logits, y):
    p = logits.softmax(-1)
    pred = p.argmax(-1)
    confusion = torch.bincount(y * 6 + pred, minlength=36).reshape(6, 6)
    f1 = (2 * confusion.diag() / (confusion.sum(0) + confusion.sum(1)).clamp_min(1)).mean()
    return {
        "accuracy": float((pred == y).float().mean()),
        "macro_f1": float(f1),
        "nll": float(decision_loss(logits, y)),
        "brier_sum": float((p - torch.nn.functional.one_hot(y, 6)).square().sum(-1).mean()),
        "ece_10_bins": expected_calibration_error(p.max(-1).values, pred == y),
        "confusion": confusion.tolist(),
    }


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def train_one(kind, seed, splits, args, out):
    torch.manual_seed(seed)
    cfg = SignalConfig(
        tuple(CHANNELS),
        EncoderConfig(
            d_model=64, n_layers=2, collapse_every=8, conv_kernel=4, collapse_cold_start=True
        ),
    )
    model = (
        SignalTacit(cfg, {"activity": Choice("activity", LABELS)})
        if kind == "tacit"
        else TemporalBaseline()
        if kind == "temporal_cnn"
        else WindowBaseline()
    ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    train_x, train_y = (t.to(args.device) for t in splits["train"])
    val_x, val_y = splits["validation"]
    val_x = val_x.to(args.device)
    best, best_state, epochs = float("inf"), None, []
    generator = torch.Generator().manual_seed(seed)
    sync(args.device)
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        indices = torch.randperm(len(train_x), generator=generator)
        total = 0.0
        for idx in indices.split(args.batch_size):
            idx = idx.to(args.device)
            loss = decision_loss(forward(model, train_x[idx]), train_y[idx], brier_weight=0.1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(idx)
        val = collect(model, val_x)
        nll = float(decision_loss(val, val_y))
        if nll < best:
            best, best_state = nll, copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
        epochs.append(
            {
                "epoch": epoch + 1,
                "train_loss": total / len(train_x),
                "validation_nll": nll,
                "validation_accuracy": metrics(val, val_y)["accuracy"],
            }
        )
        print(kind, seed, epochs[-1], flush=True)
    sync(args.device)
    train_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    model.eval()
    scaler = TemperatureScaler()
    temp_logits = collect(model, splits["temperature"][0].to(args.device))
    scaler.fit(temp_logits, splits["temperature"][1])
    policy_logits = collect(model, splits["policy"][0].to(args.device))
    policy = ConformalPolicy.fit(LABELS, scaler(policy_logits).softmax(-1), splits["policy"][1])
    # Test is first scored after training/model selection/calibration are finished.
    raw = collect(model, splits["test"][0].to(args.device))
    test_y = splits["test"][1]
    calibrated = scaler(raw)
    sets = policy.predict(calibrated.softmax(-1))
    accepted = torch.tensor([s.automated for s in sets])
    covered = [LABELS[int(y)] in s.candidates for y, s in zip(test_y, sets, strict=True)]
    result = {
        "model": kind,
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_seconds": train_seconds,
        "selected_epoch": best_epoch,
        "epochs": epochs,
        "raw": metrics(raw, test_y),
        "calibrated": metrics(calibrated, test_y),
        "temperature": float(scaler.temperature),
        "policy": asdict(policy),
        "empirical_set_coverage": sum(covered) / len(covered),
        "automation_coverage": float(accepted.float().mean()),
        "automated_accuracy": float((raw.argmax(-1)[accepted] == test_y[accepted]).float().mean())
        if accepted.any()
        else None,
        "calibration_note": "Empirical only: overlapping windows and held-out subjects are "
        "not exchangeable; no nominal conformal guarantee claimed.",
    }
    sample = splits["test"][0][:1].to(args.device)
    with torch.no_grad():
        for _ in range(10):
            forward(model, sample)
        times = []
        for _ in range(100):
            sync(args.device)
            start = time.perf_counter()
            forward(model, sample)
            sync(args.device)
            times.append((time.perf_counter() - start) * 1000)
        result["latency_ms"] = {
            "p50": float(np.median(times)),
            "p95": float(np.quantile(times, 0.95)),
            "n": len(times),
            "scope": "warm forward, batch=1, 128 samples, no preprocessing",
        }
        if kind == "tacit":
            # Stream one held-out window through irregular event boundaries.
            state, pos, chunks = None, 0, []
            for end in [3, 19, 64, 65, 128]:
                part, state = model(sample[:, pos:end], state=state)
                chunks.append(part["activity"])
                pos = end
            full = model(sample)[0]["activity"]
            result["chunk_max_logit_error"] = float((full - torch.cat(chunks, 1)).abs().max())
            result["persistent_state_tensor_bytes"] = state.nbytes
            model.temperatures[0].temperature.copy_(scaler.temperature.to(args.device))
    payload = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": asdict(cfg) if kind == "tacit" else None,
        "labels": LABELS,
        "kind": kind,
        "seed": seed,
        "temperature": float(scaler.temperature),
    }
    torch.save(payload, out / f"{kind}-seed{seed}.pt")
    np.savez_compressed(
        out / f"{kind}-seed{seed}-predictions.npz", logits=raw.numpy(), targets=test_y.numpy()
    )
    (out / f"{kind}-seed{seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["tacit", "window_mlp", "temporal_cnn"],
        default=["tacit", "window_mlp", "temporal_cnn"],
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache", type=Path, default=Path(".cache/har"))
    parser.add_argument("--output", type=Path, default=Path("results/har"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    splits, manifest = dataset(args.cache)
    manifest.update(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "torch": torch.__version__,
            "hardware": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
        }
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Partitions", manifest["split_sizes"], manifest["subjects"], flush=True)
    for kind in args.models:
        for seed in args.seeds:
            train_one(kind, seed, splits, args, args.output)


if __name__ == "__main__":
    main()
