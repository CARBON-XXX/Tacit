"""Compare S5 tracking with phase collapse enabled and disabled.

This is an exploratory, parameter-budget comparison. It does not establish
an advantage for phase collapse. Architecture, width, training budget and seed
can affect results. Run multiple seeds and inspect length generalization.
No complexity-class separation is asserted by this experiment.

    python examples/state_tracking.py
"""

from __future__ import annotations

import argparse
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from tacit import ResonantBlock
from tacit.resonance import RMSNorm


class Group:
    """The symmetric group S_k, enumerated so states can be class labels."""

    def __init__(self, k: int) -> None:
        self.k = k
        self.elements = list(itertools.permutations(range(k)))
        self.index = {p: i for i, p in enumerate(self.elements)}
        self.identity = self.index[tuple(range(k))]
        # Generators: the identity plus each adjacent transposition.
        self.generators = [tuple(range(k))]
        for i in range(k - 1):
            swapped = list(range(k))
            swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
            self.generators.append(tuple(swapped))

        # compose[g][s] = index of (generator g applied to state s)
        table = torch.empty(len(self.generators), len(self.elements), dtype=torch.long)
        for gi, g in enumerate(self.generators):
            for si, s in enumerate(self.elements):
                table[gi, si] = self.index[tuple(s[g[j]] for j in range(k))]
        self.table = table

    def __len__(self) -> int:
        return len(self.elements)

    def rollout(
        self, batch: int, length: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Random generator sequence and the running product after each step."""
        moves = torch.randint(
            len(self.generators), (batch, length), generator=generator, dtype=torch.long
        )
        state = torch.full((batch,), self.identity, dtype=torch.long)
        states = []
        for t in range(length):
            state = self.table[moves[:, t], state]
            states.append(state)
        return moves, torch.stack(states, dim=1)


class Tracker(nn.Module):
    """Embed generators, run resonant blocks, read off the running state."""

    def __init__(self, group: Group, *, d_model: int, n_layers: int, collapse_every: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(len(group.generators), d_model)
        self.blocks = nn.ModuleList(
            ResonantBlock(
                d_model,
                n_bands=4,
                collapse_every=collapse_every,
                n_phases=32,
                phase_selective=True,
                conv_kernel=0,
                n_layers=n_layers,
            )
            for _ in range(n_layers)
        )
        self.norm = RMSNorm(d_model)
        self.readout = nn.Linear(d_model, len(group))

    def forward(self, moves: torch.Tensor) -> torch.Tensor:
        h = self.embed(moves)
        for block in self.blocks:
            h = block(h)
        return self.readout(self.norm(h))


def accuracy(model: Tracker, group: Group, length: int, *, batch: int, device, seed: int) -> float:
    """Mean per-position accuracy of the predicted running state."""
    generator = torch.Generator().manual_seed(seed)
    moves, states = group.rollout(batch, length, generator)
    model.eval()
    with torch.no_grad():
        logits = model(moves.to(device))
        return float((logits.argmax(-1).cpu() == states).float().mean())


def count_params(group: Group, d_model: int, n_layers: int, collapse_every: int) -> int:
    model = Tracker(group, d_model=d_model, n_layers=n_layers, collapse_every=collapse_every)
    return sum(p.numel() for p in model.parameters())


def match_width(group: Group, target: int, args) -> int:
    """Smallest width whose collapse-free model is at least as large as ``target``.

    The collapse gate carries a symbol projection, so leaving both runs at the
    same width would hand the non-linear model several times the parameters and
    make the ablation meaningless. Widening the baseline instead errs in the
    baseline's favour.
    """
    width = args.d_model
    while count_params(group, width, args.layers, 0) < target:
        width += 8  # keep divisibility by 2 * n_bands
    return width


def run(label: str, collapse_every: int, group: Group, args, *, d_model: int) -> dict[str, float]:
    torch.manual_seed(args.seed)
    device = args.device
    model = Tracker(
        group,
        d_model=d_model,
        n_layers=args.layers,
        collapse_every=collapse_every,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps)
    generator = torch.Generator().manual_seed(args.seed)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{label}  (d_model={d_model}, {n_params / 1e6:.2f}M params)")

    model.train()
    for step in range(1, args.steps + 1):
        moves, states = group.rollout(args.batch_size, args.train_len, generator)
        logits = model(moves.to(device))
        loss = F.cross_entropy(
            logits.reshape(-1, len(group)), states.reshape(-1).to(device)
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, args.steps // 4) == 0:
            print(f"  step {step:5d}  loss {loss.detach():.4f}")

    results = {}
    for length in [args.train_len, args.train_len * 2, args.train_len * 4]:
        tag = f"len {length}" + ("" if length == args.train_len else " (extrapolated)")
        value = accuracy(
            model, group, length, batch=args.eval_batch, device=device, seed=args.seed + 99
        )
        results[tag] = value
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="track permutations of k elements")
    parser.add_argument(
        "--collapse-every",
        type=int,
        default=16,
        help="collapse period for the treatment arm; 1 collapses at every step",
    )
    parser.add_argument("--train-len", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    group = Group(args.k)
    print(
        f"S{args.k}: {len(group)} states, {len(group.generators)} generators, "
        f"trained at length {args.train_len}"
    )

    target = count_params(group, args.d_model, args.layers, args.collapse_every)
    baseline_width = match_width(group, target, args)

    treatment = f"collapse on (every {args.collapse_every})"
    with_collapse = run(treatment, args.collapse_every, group, args, d_model=args.d_model)
    without = run(
        "collapse off (linear oscillator bank)", 0, group, args, d_model=baseline_width
    )

    print("\nper-position accuracy")
    print(f"  {'':36s}" + "".join(f"{tag:>24s}" for tag in with_collapse))
    for label, result in [(treatment, with_collapse), ("collapse off", without)]:
        row = "".join(f"{value:>23.1%} " for value in result.values())
        print(f"  {label:36s}{row}")

    gap = with_collapse[f"len {args.train_len}"] - without[f"len {args.train_len}"]
    print(f"\n  chance is {1 / len(group):.1%}; the baseline is widened to match parameters")
    print(f"  gap at the training length: {gap:+.1%}")

    if args.collapse_every >= args.train_len:
        print(
            f"  note: collapse_every={args.collapse_every} at length {args.train_len} fires the\n"
            f"  non-linearity at most once per sequence, so this run barely tests it.\n"
            f"  Try --collapse-every 1."
        )
    elif abs(gap) < 0.03:
        print(
            "  check seed variability before interpreting this as 'no separation measured',\n"
            "  not as evidence either way — see the note at the top of this file."
        )
    else:
        print(
            "  before reading anything into a gap this size, re-run with a few --seed values\n"
            "  and check whether it survives."
        )


if __name__ == "__main__":
    main()
