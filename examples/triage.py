"""Train a small Tacit model to triage support tickets, then check its calibration.

Runs in a couple of minutes on a GPU and a few more on CPU. The point is not the
accuracy on a synthetic task — it is that the whole loop never generates a
token. Training is cross-entropy over the candidates each question declared, and
inference returns probabilities the caller can branch on directly.

    python examples/triage.py
    python examples/triage.py --steps 1500 --device cpu
"""

from __future__ import annotations

import argparse
import random

import torch
import torch.nn.functional as F

from tacit import (
    Boolean,
    Choice,
    EncoderConfig,
    Score,
    Tacit,
    TacitConfig,
    expected_calibration_error,
    fit_temperature,
)

QUESTIONS = {
    "urgent": Boolean(
        "Does this message need attention today?",
        when_true="the customer is blocked or names a deadline",
        when_false="a question or request with no time pressure",
    ),
    "team": Choice(
        "Which team should handle this?",
        {
            "billing": "payments, invoices, refunds, payouts",
            "technical": "bugs, outages, errors, integrations",
            "sales": "pricing, plans, upgrades, new accounts",
        },
    ),
    "anger": Score(
        "How frustrated does the customer sound?",
        ["calm", "annoyed", "furious"],
    ),
}

TOPICS = {
    "billing": [
        "my invoice {n} was charged twice",
        "the refund for order {n} never arrived",
        "payouts have been failing for {n} days",
        "I was billed {n} dollars I do not recognise",
    ],
    "technical": [
        "the API returns a {n}0{n} error on every call",
        "webhooks stopped firing about {n} hours ago",
        "the dashboard has been down for {n} minutes",
        "our integration broke after release {n}",
    ],
    "sales": [
        "what does the team plan cost for {n} seats",
        "we want to upgrade {n} accounts next quarter",
        "can you share pricing for {n} users",
        "is there a discount for a {n} year contract",
    ],
}

URGENT_MARKS = ["we are blocked", "this is production", "customers cannot pay", "deadline is today"]
TONE_MARKS = [
    ["whenever you get a chance", "no rush at all", "just checking in"],
    ["this is the second time", "still waiting on this", "can someone look"],
    ["this is completely unacceptable", "we are cancelling", "absolutely furious"],
]

# Some tickets carry no signal for a given question. The label is still drawn,
# so no model can recover it — which is the point. A useful decision model is
# confident on `team`, where the body settles it, and hedges on the rest. A
# model that is confident everywhere is the failure this demo is built to show.
P_TONE_HIDDEN = 0.35
P_URGENCY_HIDDEN = 0.30

# Best accuracy any model can reach, given the labels that were hidden above.
#   team   the body always settles it
#   anger  seen tone is decisive; hidden tone leaves a 1-in-3 guess
#   urgent a marker proves urgency; without one, "no" is right 0.15/0.65 short
CEILING = {
    "team": 1.0,
    "anger": (1 - P_TONE_HIDDEN) + P_TONE_HIDDEN / 3,
    "urgent": 0.5 * (1 - P_URGENCY_HIDDEN) + (1 - 0.5 * (1 - P_URGENCY_HIDDEN)) * (
        1 - (0.5 * P_URGENCY_HIDDEN) / (1 - 0.5 * (1 - P_URGENCY_HIDDEN))
    ),
}


def make_example(rng: random.Random) -> tuple[str, dict[str, int]]:
    """One ticket plus the gold candidate index for each question."""
    team = rng.choice(list(TOPICS))
    body = rng.choice(TOPICS[team]).format(n=rng.randint(2, 9))
    parts = [body]

    anger = rng.randrange(3)
    if rng.random() > P_TONE_HIDDEN:
        parts.append(rng.choice(TONE_MARKS[anger]))

    urgent = rng.random() < 0.5
    if urgent and rng.random() > P_URGENCY_HIDDEN:
        parts.append(rng.choice(URGENT_MARKS))

    rng.shuffle(parts)
    text = ", ".join(parts) + "."
    # Boolean labels are ["true", "false"], so index 0 means urgent.
    return text, {"urgent": 0 if urgent else 1, "team": list(TOPICS).index(team), "anger": anger}


def make_split(n: int, seed: int) -> list[tuple[str, dict[str, int]]]:
    rng = random.Random(seed)
    return [make_example(rng) for _ in range(n)]


def batch_logits(model: Tacit, texts: list[str], *, calibrated: bool) -> dict[str, torch.Tensor]:
    state = model.encode_texts(texts)
    return {
        name: model.score(state, q, calibrated=calibrated) for name, q in QUESTIONS.items()
    }


def pool_across_questions(per_question: dict[str, tuple[torch.Tensor, torch.Tensor]]):
    """Stack logits from questions with different candidate counts.

    The head carries a single temperature, so it has to be fitted against a
    sample drawn from every question it serves — fitting on one question and
    applying the result to the rest is how a calibration step makes things
    worse. Short rows are padded with zeros and masked out.
    """
    width = max(v.shape[1] for v, _ in per_question.values())
    logits, targets, masks = [], [], []
    for value, gold in per_question.values():
        n = value.shape[1]
        padded = F.pad(value, (0, width - n), value=0.0)
        mask = torch.zeros(value.shape[0], width, dtype=torch.bool)
        mask[:, :n] = True
        logits.append(padded)
        targets.append(gold)
        masks.append(mask)
    return torch.cat(logits), torch.cat(targets), torch.cat(masks)


def collect(model: Tacit, split, batch_size: int, *, calibrated: bool):
    """Run a split and return per-question (logits, targets) on the CPU."""
    model.eval()
    out: dict[str, tuple[list, list]] = {name: ([], []) for name in QUESTIONS}
    with torch.no_grad():
        for i in range(0, len(split), batch_size):
            chunk = split[i : i + batch_size]
            scored = batch_logits(model, [t for t, _ in chunk], calibrated=calibrated)
            for name, value in scored.items():
                out[name][0].append(value.cpu())
                out[name][1].append(torch.tensor([labels[name] for _, labels in chunk]))
    return {name: (torch.cat(v), torch.cat(g)) for name, (v, g) in out.items()}


def report(per_question) -> tuple[dict[str, float], float]:
    """Per-question accuracy and one pooled expected calibration error."""
    accuracy, confidences, correctness = {}, [], []
    for name, (value, gold) in per_question.items():
        probs = value.softmax(-1)
        top = probs.argmax(-1)
        accuracy[name] = float((top == gold).float().mean())
        confidences.append(probs.max(-1).values)
        correctness.append(top == gold)
    return accuracy, expected_calibration_error(
        torch.cat(confidences), torch.cat(correctness)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--eval-size", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train = make_split(args.train_size, seed=args.seed)
    calib = make_split(args.eval_size, seed=args.seed + 1)
    test = make_split(args.eval_size, seed=args.seed + 2)

    model = Tacit(
        TacitConfig(
            encoder=EncoderConfig(
                d_model=args.d_model,
                n_layers=args.layers,
                collapse_every=16,
                n_phases=16,
            )
        )
    ).to(args.device)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params on {args.device}")
    print(f"sample ticket: {train[0][0]!r}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps)
    rng = random.Random(args.seed)

    model.train()
    for step in range(1, args.steps + 1):
        chunk = [train[rng.randrange(len(train))] for _ in range(args.batch_size)]
        texts = [t for t, _ in chunk]
        # calibrated=False: the temperature is fitted afterwards on held-out
        # data, not learned jointly with everything else.
        logits = batch_logits(model, texts, calibrated=False)
        loss = sum(
            F.cross_entropy(
                value,
                torch.tensor([labels[name] for _, labels in chunk], device=value.device),
            )
            for name, value in logits.items()
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 200 == 0 or step == 1:
            print(f"step {step:5d}  loss {loss.detach():.4f}")

    print()
    accuracy, ece_before = report(collect(model, test, args.batch_size, calibrated=False))
    for name, value in accuracy.items():
        print(f"  {name:8s} accuracy {value:6.1%}   (ceiling {CEILING[name]:.1%})")

    logits, targets, mask = pool_across_questions(
        collect(model, calib, args.batch_size, calibrated=False)
    )
    temperature = fit_temperature(model.head, logits, targets, candidate_mask=mask)
    model.clear_cache()

    _, ece_after = report(collect(model, test, args.batch_size, calibrated=True))
    print(f"\n  expected calibration error {ece_before:.4f} -> {ece_after:.4f}"
          f"  (fitted temperature {temperature:.3f})")

    print("\ntyped answers on a held-out ticket")
    text = test[0][0]
    print(f"  state: {text!r}")
    for name, answer in model.evaluate(text, QUESTIONS).items():
        print(f"    {name:8s} {answer}")

    print("\nthe same model, used as a session (state carries across turns)")
    session = model.session()
    for turn in [
        "customer: payouts have been failing for three days, we are blocked",
        "agent: we pushed a fix, can you retry the payout now?",
        "customer: it went through, thanks",
    ]:
        session.observe(turn)
        answer = session.ask({"urgent": QUESTIONS["urgent"]})["urgent"]
        print(f"    after {session.bytes_seen:4d} bytes  urgent p={answer.probability:.3f}")


if __name__ == "__main__":
    main()
