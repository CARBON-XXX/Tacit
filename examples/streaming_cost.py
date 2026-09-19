"""Measure what carrying state actually buys, and where it starts to pay.

A stateless decision endpoint has to be handed the whole history on every call,
so its cost per turn grows with the conversation. A session folds each turn into
a fixed-size complex state and never looks at the text again, so its cost per
turn is flat.

That is an asymptotic claim, and asymptotics are not free. The streaming path
here is plain PyTorch with one kernel launch per byte, so its constant is large;
re-reading a short transcript in one parallel pass is genuinely faster. This
script finds the crossover on your machine instead of asserting one.

    python examples/streaming_cost.py
    python examples/streaming_cost.py --turns 100 --device cpu
"""

from __future__ import annotations

import argparse
import time

import torch

from tacit import Boolean, EncoderConfig, Tacit, TacitConfig

TURN = "customer: the payout failed again and we are still blocked on this. "


def timed(fn, *, device: str, repeats: int = 1) -> float:
    """Wall-clock milliseconds, synchronising around CUDA work."""
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / repeats


def state_bytes(session) -> int:
    total = 0
    for wavefield, conv_buffer in session._encoder_state.blocks:
        for tensor in (wavefield, conv_buffer):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = Tacit(
        TacitConfig(encoder=EncoderConfig(d_model=args.d_model, n_layers=args.layers))
    ).eval().to(args.device)
    print(
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params on {args.device}, "
        f"{len(TURN)} bytes per turn\n"
    )

    session = model.session()
    transcript = ""
    crossover = None
    rows = []

    for turn in range(1, args.turns + 1):
        transcript += TURN
        fold_ms = timed(lambda: session.observe(TURN), device=args.device)
        reread_ms = timed(
            lambda whole=transcript: model.encode_texts([whole]), device=args.device
        )
        rows.append((turn, len(transcript), fold_ms, reread_ms))
        if crossover is None and reread_ms > fold_ms:
            crossover = turn

    print(f"{'turns':>6} {'transcript':>11} {'fold (ms)':>11} {'re-read (ms)':>13}")
    for turn, size, fold_ms, reread_ms in rows:
        if turn % max(1, args.turns // 10) == 0 or turn == crossover:
            marker = "  <- crossover" if turn == crossover else ""
            print(f"{turn:6d} {size:10d}B {fold_ms:11.1f} {reread_ms:13.1f}{marker}")

    last = rows[-1]
    print(
        f"\ncarried state is {state_bytes(session)} bytes and does not grow; "
        f"the transcript reached {last[1]} bytes"
    )
    if crossover is None:
        print("re-reading never got slower than folding at this length — raise --turns")
    else:
        print(
            f"folding wins from turn {crossover} onward; by turn {last[0]} it is "
            f"{last[3] / last[2]:.1f}x cheaper per turn"
        )
    print(
        "\nfolding is flat because the state is fixed-size. It is not faster in "
        "absolute terms at short lengths: the streaming path launches one kernel "
        "per byte, which is a constant a fused kernel would remove."
    )

    answer = session.ask({"blocked": Boolean("Is the customer still blocked?")})["blocked"]
    print(f"\nanswering from state alone, no transcript re-read: {answer}")


if __name__ == "__main__":
    main()
