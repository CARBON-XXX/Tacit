"""Rescore, compare and export pinned Jev/Laya/Tacit predictions without secrets."""

import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np
from general_data import load_corpus
from publish_typed import export_predictions, sha256, write
from typed_common import labels, load_cases, summarize


def bootstrap_cluster(case):
    """Do not count paired layouts or duplicate input text as independent cases."""
    if case["id"].startswith("mnli/"):
        state = case["state"]
        premise = state["premise"] if isinstance(state, dict) else state
        return hashlib.sha256(" ".join(premise.split()).encode()).hexdigest()
    if case["id"].startswith(("news/", "emotion/")):
        return hashlib.sha256(" ".join(case["state"].lower().split()).encode()).hexdigest()
    return case["id"]


def paired_clusters(candidate, reference, cases, samples=10000):
    def key(r):
        return r["id"], r["question"]

    a, b = {key(r): r for r in candidate}, {key(r): r for r in reference}
    if len(a) != len(candidate) or len(b) != len(reference) or a.keys() != b.keys():
        raise ValueError("paired predictions must match one-to-one")
    grouped = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for k, r in a.items():
        other = b[k]
        if any(r[f] != other[f] for f in ["labels", "gold_index", "target", "workflow", "type"]):
            raise ValueError("paired gold or schema differs")
        case = cases[r["id"]]
        group = bootstrap_cluster(case)
        hit = int(np.argmax(r["probabilities"]) == r["gold_index"])
        hit -= int(np.argmax(other["probabilities"]) == r["gold_index"])
        value = grouped[r["workflow"]][group]
        value[0] += hit
        value[1] += 1
    rng = np.random.default_rng(2026)
    numer, denom = np.zeros(samples), np.zeros(samples)
    total, count, clusters = 0, 0, 0
    for workflow in sorted(grouped):
        rows = np.asarray([v for _, v in sorted(grouped[workflow].items())])
        selected = rows[rng.integers(0, len(rows), size=(samples, len(rows)))]
        numer += selected[:, :, 0].sum(1)
        denom += selected[:, :, 1].sum(1)
        total += rows[:, 0].sum()
        count += rows[:, 1].sum()
        clusters += len(rows)
    return {
        "accuracy_difference": float(total / count),
        "ci95": np.quantile(numer / denom, [0.025, 0.975]).tolist(),
        "clusters": clusters,
        "decisions": int(count),
        "resamples": samples,
        "seed": 2026,
        "unit": (
            "workflow-stratified case; shared NLI premises/layouts and duplicate auxiliary "
            "input texts remain together"
        ),
        "scope": (
            "fixed checkpoints, exploratory; no training-seed uncertainty "
            "or multiple-test correction"
        ),
    }


def main():
    destination = Path("results/comparison-snapshot")
    destination.mkdir(exist_ok=True)
    cases = {c["id"]: c for c in load_corpus(".cache/general-v1", "test")}
    if {c["id"] for c in load_cases(split="test")} != {
        k for k in cases if not k.startswith("mnli/")
    }:
        raise ValueError("workflow test IDs changed")
    paths = {
        "tacit-mixed": Path("results/general/tacit-sdk-test.json"),
        "laya-general": Path("results/general/laya-general-test.json"),
        "jev-typed": Path("results/typed/jev-official-test.json"),
        "jev-nli": Path("results/general/jev-official-test.json"),
    }
    predictions, reports, interface_audits = {}, {}, {}
    for name, path in paths.items():
        data = json.loads(path.read_text())
        predictions[name] = data.pop("records")
        rebuilt = export_predictions(destination, name, predictions[name], cases)
        expected = data["metrics"]
        if "overall" in expected:
            if rebuilt != expected:
                raise ValueError("metric mismatch on export: " + name)
        else:
            for family in ["nli", "typed"]:
                records = [
                    r for r in predictions[name] if r["id"].startswith("mnli/") == (family == "nli")
                ]
                if summarize(records) != expected[family]:
                    raise ValueError("family metric mismatch on export: " + name)
        reports[name] = data
        write(destination / (name + ".json"), data)
        if name.startswith("jev-"):
            choices, mismatches, emitted_hits, probability_hits = 0, [], 0, 0
            with (destination / (name + ".responses.jsonl")).open("w") as stream:
                for line in path.with_suffix(".jsonl").read_text().splitlines():
                    raw = json.loads(line)
                    # Whitelist provider response fields; never publish headers/account data.
                    raw["response"] = {k: raw["response"][k] for k in ["model", "answers", "usage"]}
                    stream.write(json.dumps(raw, separators=(",", ":")) + "\n")
                    for question, answer in raw["response"]["answers"].items():
                        if answer["type"] != "choice":
                            continue
                        case = cases[raw["id"]]
                        keys = labels(case["questions"][question])
                        predicted = keys[int(np.argmax([answer["probabilities"][k] for k in keys]))]
                        gold = case["gold"][question]["label"]
                        choices += 1
                        emitted_hits += answer["choice"] == gold
                        probability_hits += predicted == gold
                        if predicted != answer["choice"]:
                            mismatches.append({"id": raw["id"], "question": question})
            interface_audits[name] = {
                "choice_decisions": choices,
                "returned_choice_accuracy": emitted_hits / choices,
                "probability_argmax_accuracy": probability_hits / choices,
                "choice_field_differs_from_argmax": mismatches,
                "primary_protocol": "all-model accuracy uses exposed probability argmax",
            }
    paired = json.loads(Path("results/typed/paired-refined-laya.json").read_text())
    for model, name in [("tacit", "tacit-refined"), ("laya", "laya-typed")]:
        predictions[name] = paired["records"][model]
        metrics = export_predictions(destination, name, predictions[name], cases)
        if metrics != paired["metrics"][model]:
            raise ValueError("paired metrics changed")
    paired.pop("records")
    write(destination / "paired-refined.json", paired)
    for name, directory in [
        ("tacit-refined", "results/typed/forked-refine-seed0"),
        ("tacit-mixed", "results/general/mixed-seed0"),
    ]:
        run = Path(directory)
        data = json.loads((run / "result.json").read_text())
        data.pop("records")
        data["manifest"] = json.loads((run / "manifest.json").read_text())
        data["history"] = json.loads((run / "history.json").read_text())
        data["checkpoint_sha256"] = sha256(run / "best.pt")
        data["checkpoint_bytes"] = (run / "best.pt").stat().st_size
        write(destination / (name + ".training.json"), data)
    typed_rows, nli_rows = {}, {}
    for name, rows in predictions.items():
        typed = [r for r in rows if not r["id"].startswith("mnli/")]
        nli = [r for r in rows if r["id"].startswith("mnli/")]
        if typed:
            typed_rows[name] = summarize(typed)["overall"]
        if nli:
            nli_rows[name] = {
                kind: summarize([r for r in nli if r["type"] == kind])["overall"]
                for kind in ["choice", "noul"]
            }
    pairs = {}
    for baseline in ["laya-typed", "jev-typed"]:
        pairs["workflow/refined-minus-" + baseline] = paired_clusters(
            predictions["tacit-refined"], predictions[baseline], cases
        )
    for baseline in ["laya-general", "jev-nli"]:
        for kind in ["choice", "noul"]:

            def select(name, kind=kind):
                return [
                    r
                    for r in predictions[name]
                    if r["id"].startswith("mnli/") and r["type"] == kind
                ]

            pairs["nli/" + kind + "/mixed-minus-" + baseline] = paired_clusters(
                select("tacit-mixed"), select(baseline), cases
            )
    nli_splits = {}
    for name in ["tacit-mixed", "laya-general", "jev-nli"]:
        nli_splits[name] = {}
        for split in ["validation_matched", "validation_mismatched"]:
            rows = [
                r
                for r in predictions[name]
                if r["id"].startswith("mnli/" + split + "/") and r["type"] == "choice"
            ]
            nli_splits[name][split] = summarize(rows)["overall"]
    summary = {
        "status": "broad superiority to Jev/Laya not established",
        "workflow": typed_rows,
        "nli": nli_rows,
        "nli_relation_by_official_split": nli_splits,
        "paired_accuracy": pairs,
        "paired_workflow_latency": paired["latency"],
        "jev_interface_audit": interface_audits,
        "api_cost_usd": str(
            sum(
                (Decimal(reports[name]["run_cost_usd"]) for name in ["jev-typed", "jev-nli"]),
                Decimal(0),
            )
        ),
        "cumulative_budget": reports["jev-nli"]["cumulative_budget"],
        "cost_basis": "reported input tokens times published $0.042/M; not invoice reconciliation",
        "limitations": [
            "One Tacit seed; supervised NLI and workflow training, not arbitrary-task proof.",
            "Workflow gold is a synthetic teacher distribution; NLI gold is human-labeled.",
            "1000 matched + 1000 mismatched MultiNLI cases; not the authors' XNLI protocol.",
            "Hypothesis in instructions; Tacit trained on this format. Other formats untested.",
            "Laya general and typed checkpoints are distinct; both are reported.",
            "Jev rounds to 2 decimals; renormalize. Zeros inflate NLL with the 1e-12 floor.",
            "Returned Jev choice may differ from exposed argmax; both are audited.",
            "Network Jev latency is not directly comparable to local GPU SDK timing.",
            "Mixed Tacit BF16 batch/SDK: 71.15%/71.10% on workflows, 3 argmax differences.",
            "NLI batch and SDK predicted labels agree; SDK measurements are primary here.",
            "Public-test iterations are exploratory; fresh final tests and more seeds needed.",
        ],
    }
    write(destination / "summary.json", summary)
    write(
        destination / "data-audit.json",
        json.loads(Path("results/general/data-audit.json").read_text()),
    )
    write(
        destination / "corpus-manifest.json",
        json.loads(Path(".cache/general-v1/manifest.json").read_text()),
    )
    write(
        destination / "checksums.json",
        {
            p.name: sha256(p)
            for p in sorted(destination.iterdir())
            if p.is_file() and p.name != "checksums.json"
        },
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ["workflow", "nli", "nli_relation_by_official_split", "limitations"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
