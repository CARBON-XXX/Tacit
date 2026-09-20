# Launch copy (publish after the repository is live)

Jev got me thinking: what if typed decisions had recurrent memory?

Meet Tacit: a small PyTorch experiment for stateful typed decisions.

observations → persistent state → boolean / choice / score

Code + training demos below. No pretrained weights yet. 🧵

---

Our text-generation experiments did not work out as hoped. Instead of claiming
that was secretly a success, we changed the question: can the same core support
useful decisions over a stream of observations?

---

Tacit carries a fixed-size complex recurrent state between observations.
Questions return distributions over declared options.

Fixed-size state is lossy memory, not unlimited recall. The key experiment is
whether it retains what a downstream decision actually needs.

---

Included: synthetic ticket triage, held-out temperature fitting, a streaming
cost comparison, and a phase-collapse ablation.

No established collapse advantage. No Jev performance comparison. This is a
research starting point, with code you can run and challenge.

---

https://github.com/CARBON-XXX/Tacit

MIT · PyTorch · train-from-scratch examples

Inspired by Jev's typed-decision framing:
https://typesafe.ai/blog/introducing-system-one-models-and-jev
