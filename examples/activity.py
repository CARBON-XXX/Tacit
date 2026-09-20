"""Load a trained numeric decision model; classify one normalized sensor window.

First run benchmarks/har.py. With no --window, use the first official test window
from that script's local dataset cache. Supply a [128, 9] raw-signal .npy file to
classify a different window in the same channel order and physical units.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tacit import Choice, ConformalPolicy, EncoderConfig, SignalConfig, SignalTacit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("results/har/tacit-seed0.pt"))
    parser.add_argument("--window", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    folder = args.checkpoint.parent
    manifest = json.loads((folder / "manifest.json").read_text())
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = dict(saved["config"])
    config["encoder"] = EncoderConfig(**config["encoder"])
    model = SignalTacit(SignalConfig(**config), {"activity": Choice("activity", saved["labels"])})
    model.load_state_dict(saved["state_dict"])
    model.to(args.device).eval()
    if args.window:
        window = np.load(args.window, allow_pickle=False)
    else:
        with np.load(".cache/har/raw-windows.npz") as cache:
            window = cache["test_x"][0]
    if window.shape != (128, 9):
        raise ValueError("expected a [128, 9] sensor window")
    values = torch.as_tensor(window, dtype=torch.float32)
    values = (values - torch.tensor(manifest["mean"])) / torch.tensor(manifest["std"])
    answers, state = model.evaluate(values[None])
    answer = answers[0]["activity"]
    result = json.loads(args.checkpoint.with_suffix(".json").read_text())
    policy_data = dict(result["policy"])
    policy_data["labels"] = tuple(policy_data["labels"])
    policy = ConformalPolicy(**policy_data)
    prediction = policy.predict(torch.tensor([[answer.probabilities[k] for k in saved["labels"]]]))[
        0
    ]
    print(
        json.dumps(
            {
                "activity": answer.value,
                "probabilities": answer.probabilities,
                "prediction_set": prediction.candidates,
                "automated": prediction.automated,
                "state_tensor_bytes": state.nbytes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
