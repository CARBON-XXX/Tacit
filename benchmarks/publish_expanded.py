"""Publish complete expanded reference tests and paired input-format sensitivity.

This snapshot evaluates the previously frozen mixed Tacit checkpoint. It never
reads the running expanded-training model or presents validation as test scores.
"""

import copy
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
from gateway_budget import BudgetLedger
from general_data import load_corpus
from publish_comparison import paired_clusters
from publish_typed import export_predictions, sha256, write
from train_general_v2 import measurements
from typed_common import labels, summarize


def layout_comparison(records, cases, kind):
    canonical = [
        r
        for r in records
        if r["id"].startswith("mnli/") and not r["id"].endswith("/structured") and r["type"] == kind
    ]
    structured = [
        copy.deepcopy(r) for r in records if r["id"].endswith("/structured") and r["type"] == kind
    ]
    for row in structured:
        row["id"] = row["id"].removesuffix("/structured")
    return paired_clusters(structured, canonical, cases)


def main():
    destination = Path("results/expanded-snapshot")
    destination.mkdir(exist_ok=True)
    cases = {c["id"]: c for c in load_corpus(".cache/general-v2", "test")}
    paths = {
        "tacit-mixed": [
            "results/general/tacit-sdk-test.json",
            "results/general/tacit-mixed-expanded-test.json",
        ],
        "laya-general": [
            "results/general/laya-general-test.json",
            "results/general/laya-expanded-test.json",
        ],
        "jev": [
            "results/typed/jev-official-test.json",
            "results/general/jev-official-test.json",
            "results/general/jev-expanded-test.json",
        ],
    }
    predictions, metrics, audits, costs = {}, {}, {}, []
    for name, sources in paths.items():
        reports, records = [], []
        for source in sources:
            report = json.loads(Path(source).read_text())
            rows = report.pop("records")
            records.extend(rows)
            reports.append(report)
            if name == "jev":
                costs.append(Decimal(report["run_cost_usd"]))
        expected = {(c["id"], q) for c in cases.values() for q in c["questions"]}
        observed = [(r["id"], r["question"]) for r in records]
        if len(observed) != len(expected) or set(observed) != expected:
            raise ValueError("missing or duplicate decisions: " + name)
        if name != "jev" and len({r["checkpoint_sha256"] for r in reports}) != 1:
            raise ValueError("one reported model cannot combine different checkpoints")
        if export_predictions(destination, name, records, cases) != summarize(records):
            raise ValueError("canonical rescore differs")
        predictions[name] = records
        metrics[name] = measurements(records)
        write(destination / (name + ".json"), {"runs": reports, "tracks": metrics[name]})
        if name == "jev":
            audit = {}
            with (destination / "jev.responses.jsonl").open("w") as stream:
                for source in sources:
                    for line in Path(source).with_suffix(".jsonl").read_text().splitlines():
                        row = json.loads(line)
                        response = {k: row["response"][k] for k in ("model", "answers", "usage")}
                        stream.write(
                            json.dumps(
                                {
                                    "id": row["id"],
                                    "request_sha256": row["request_sha256"],
                                    "response": response,
                                },
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        for question, answer in response["answers"].items():
                            if answer["type"] != "choice":
                                continue
                            case = cases[row["id"]]
                            family = (
                                "nli/structured"
                                if row["id"].endswith("/structured")
                                else (
                                    "nli/text"
                                    if row["id"].startswith("mnli/")
                                    else (
                                        row["id"].split("/")[0]
                                        if row["id"].startswith(("news/", "emotion/"))
                                        else "typed"
                                    )
                                )
                            )
                            item = audit.setdefault(
                                family,
                                {
                                    "decisions": 0,
                                    "returned_hits": 0,
                                    "argmax_hits": 0,
                                    "mismatches": [],
                                },
                            )
                            keys = labels(case["questions"][question])
                            argmax = keys[
                                int(np.argmax([answer["probabilities"][k] for k in keys]))
                            ]
                            gold = case["gold"][question]["label"]
                            item["decisions"] += 1
                            item["returned_hits"] += answer["choice"] == gold
                            item["argmax_hits"] += argmax == gold
                            if argmax != answer["choice"]:
                                item["mismatches"].append({"id": row["id"], "question": question})
            audits[name] = audit
    pairs = {}
    for track in metrics["tacit-mixed"]:

        def select(name, track=track):
            from train_general_v2 import track as track_name

            return [r for r in predictions[name] if track_name(r) == track]

        for reference in ("jev", "laya-general"):
            pairs[track + "/mixed-minus-" + reference] = paired_clusters(
                select("tacit-mixed"), select(reference), cases
            )
    private = Path.home() / ".config/tacit/jev-budget.json"
    if private.exists():
        ledger = BudgetLedger(private)
        try:
            budget = ledger.summary()
        finally:
            ledger.close()
        budget["snapshot"] = "cumulative task ledger at export; includes prior reservations"
    else:
        budget = json.loads(Path(paths["jev"][-1]).read_text())["cumulative_budget"]
        budget["snapshot"] = "historical budget snapshot saved with the last evaluated run"
    summary = {
        "status": "previous mixed checkpoint; expanded training still separate; no broad win",
        "cases": len(cases),
        "decisions_per_model": len(predictions["jev"]),
        "tracks": {
            name: {k: v["overall"] for k, v in rows.items()} for name, rows in metrics.items()
        },
        "paired_accuracy": pairs,
        "structured_minus_text_layout_accuracy": {
            name: {kind: layout_comparison(rows, cases, kind) for kind in ("choice", "noul")}
            for name, rows in predictions.items()
        },
        "jev_choice_audit": audits["jev"],
        "successful_api_cost_usd": str(sum(costs, Decimal(0))),
        "budget": budget,
        "limitations": [
            "A single mixed Tacit checkpoint across all tracks, frozen before the expanded run.",
            "Published workflow-specialist Laya scores remain separate; this is general Laya.",
            "Paired layouts are the same NLI cases, not independent additional examples.",
            "Emotion labels are machine-generated and research-use restricted.",
            "Jev zero probabilities affect exposed-output NLL with the common 1e-12 floor.",
            "Expanded local evaluations ran during training: accuracy only, no latency claim.",
            "Fixed-checkpoint bootstrap; no seed uncertainty or multiple-test correction.",
            "Expanded checkpoint test results are not available in this snapshot.",
        ],
    }
    write(destination / "summary.json", summary)
    write(
        destination / "corpus-manifest.json",
        json.loads(Path(".cache/general-v2/manifest.json").read_text()),
    )
    write(
        destination / "data-audit.json",
        json.loads(Path("results/general/expanded-data-audit.json").read_text()),
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
                name: {k: v["accuracy"] for k, v in rows.items()}
                for name, rows in summary["tracks"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
