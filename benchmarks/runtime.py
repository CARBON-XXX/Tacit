"""Same-model runtime regression: byte steps versus resumable chunks.

No competitor speed ratio: hardware, inputs and decision quality must match.
Random weights test runtime/structural invariants, not decision accuracy.
Run after stopping other GPU jobs: PYTHONPATH=src python benchmarks/runtime.py
"""

import json
import random
import time
from pathlib import Path

import torch

from tacit import Boolean, Choice, EncoderConfig, Tacit, TacitConfig


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Tacit(TacitConfig(encoder=EncoderConfig(d_model=192, n_layers=3))).to(device).eval()

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def measure_pair(first, second, repeats=30):
        functions, samples = [first, second], [[], []]
        rng = random.Random(0)
        for _ in range(10):
            for fn in functions:
                fn()
        for _ in range(repeats):
            order = [0, 1]
            rng.shuffle(order)
            for i in order:
                sync()
                start = time.perf_counter()
                functions[i]()
                sync()
                samples[i].append(1000 * (time.perf_counter() - start))
        result = []
        for times in samples:
            v = torch.tensor(times)
            result.append(
                {
                    "p50_ms": float(v.quantile(0.5)),
                    "p95_ms": float(v.quantile(0.95)),
                    "samples_ms": times,
                }
            )
        return result

    result = {
        "hardware": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        "torch": torch.__version__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "note": "Same randomly initialized model; runtime and numerical parity only. "
        "No language capability, Jev or Laya ranking. Paired timing with seeded AB/BA order.",
        "absorb": [],
        "heads": [],
    }
    with torch.no_grad():
        # Resume at a non-aligned position, exercising the production event path.
        _, initial = model.encoder.absorb(
            torch.randint(0, 256, (1, 7), device=device), model.encoder.init_state(1, device)
        )
        for n in [32, 128, 512]:
            tokens = torch.randint(0, 256, (1, n), device=device)

            def legacy(n=n, tokens=tokens):
                state = initial
                for i in range(n):
                    h, state = model.encoder.step(tokens[:, i], state)
                return h[:, 0], state

            def chunked(tokens=tokens):
                return model.encoder.absorb(tokens, initial)

            a, sa = legacy()
            b, sb = chunked()
            old_time, new_time = measure_pair(legacy, chunked)
            result["absorb"].append(
                {
                    "bytes": n,
                    "legacy": old_time,
                    "chunked": new_time,
                    "hidden_max_error": float((a - b).abs().max()),
                    "state_max_error": max(
                        float((x[0] - y[0]).abs().max())
                        for x, y in zip(sa.blocks, sb.blocks, strict=True)
                    ),
                    "persistent_storage_bytes": sum(
                        t.untyped_storage().nbytes()
                        for pair in sb.blocks
                        for t in pair
                        if t is not None
                    ),
                }
            )
        state = model.encode_texts(["An event is waiting for a decision."])
        for count in [1, 3, 14, 32]:
            questions = {str(i): Boolean(f"Does event flag {i} apply?") for i in range(count)}
            model.score_many(state, questions)

            def serial(questions=questions):
                return {k: model.score(state, q) for k, q in questions.items()}

            def batched(questions=questions):
                return model.score_many(state, questions)

            first, second = serial(), batched()
            old_time, new_time = measure_pair(serial, batched, 200)
            result["heads"].append(
                {
                    "questions": count,
                    "serial": old_time,
                    "batched": new_time,
                    "logit_max_error": max(
                        float((first[k] - second[k]).abs().max()) for k in first
                    ),
                }
            )
        options = [f"route_{i}" for i in range(255)]
        first = model.score(state, Choice("Select a route.", options))
        reverse = model.score(state, Choice("Select a route.", list(reversed(options))))
        result["255_candidates"] = {
            "permutation_max_logit_error": float((first - reverse.flip(-1)).abs().max()),
            "finite": bool(torch.isfinite(first).all()),
            "accuracy": None,
        }
    path = Path("results/runtime.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)
    for row in result["absorb"]:
        print(row["bytes"], row["legacy"]["p50_ms"], row["chunked"]["p50_ms"])


if __name__ == "__main__":
    main()
