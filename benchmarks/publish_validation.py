"""Export complete development references; never score or publish the fresh gate."""

import hashlib
import json
from decimal import Decimal
from pathlib import Path

from jev_official import MODEL, answer_records, token_cost
from jev_validation import validation_cases
from publish_typed import export_predictions, sha256, write
from train_general_v2 import measurements
from typed_common import record, summarize


def main():
    cases, manifest = validation_cases(".cache/general-v2")
    cases = {c["id"]: c for c in cases}
    expected = {(c["id"], q) for c in cases.values() for q in c["questions"]}
    destination = Path("results/validation-snapshot")
    destination.mkdir(exist_ok=True)
    metrics, reports, jev_predictions = {}, {}, {}
    for name, source in {
        "jev": "results/general/jev-validation.json",
        "laya-general": "results/general/laya-validation.json",
    }.items():
        report = json.loads(Path(source).read_text())
        if not report["evaluation_split"].startswith("validation"):
            raise ValueError("only validation reports belong in this snapshot")
        if report["corpus_manifest"] != manifest:
            raise ValueError("reference inputs differ")
        rows = report.pop("records")
        observed = [(r["id"], r["question"]) for r in rows]
        if len(observed) != len(expected) or set(observed) != expected:
            raise ValueError("incomplete or duplicate reference decisions")
        rebuilt = [record(cases[r["id"]], r["question"], r["probabilities"]) for r in rows]
        if export_predictions(destination, name, rows, cases) != summarize(rebuilt):
            raise ValueError("prediction export changed metrics")
        rescored = measurements(rebuilt)
        for track, values in rescored.items():
            for field in ("accuracy", "soft_nll", "brier_sum"):
                difference = values["overall"][field] - report["tracks"][track]["overall"][field]
                if abs(difference) > 1e-10:
                    raise ValueError("reference metric changed during canonical rescore")
        metrics[name] = {k: v["overall"] for k, v in rescored.items()}
        reports[name] = report
        if name == "jev":
            jev_predictions = {(r["id"], r["question"]): r for r in rows}
        write(destination / (name + ".json"), report)
    seen, cost, tokens = set(), Decimal(0), 0
    with (destination / "jev.responses.jsonl").open("w") as output:
        for line in Path("results/general/jev-validation.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["id"] not in cases or row["id"] in seen:
                raise ValueError("unexpected or duplicate raw reference response")
            seen.add(row["id"])
            case = cases[row["id"]]
            payload = {"model": MODEL, "state": case["state"], "questions": case["questions"]}
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            if hashlib.sha256(encoded).hexdigest() != row["request_sha256"]:
                raise ValueError("raw response belongs to different inputs")
            response = {k: row["response"][k] for k in ("model", "answers", "usage")}
            for reconstructed in answer_records(case, response):
                saved = jev_predictions[(reconstructed["id"], reconstructed["question"])]
                if reconstructed["probabilities"] != saved["probabilities"]:
                    raise ValueError("Jev prediction differs from the exposed raw probabilities")
            cost += token_cost(response)
            tokens += response["usage"]["input_tokens"]
            output.write(
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
    if seen != set(cases) or cost != Decimal(reports["jev"]["run_cost_usd"]):
        raise ValueError("raw response coverage or billing mismatch")
    if tokens != reports["jev"]["input_tokens"]:
        raise ValueError("reported token count changed")
    write(destination / "corpus-manifest.json", manifest)
    write(
        destination / "summary.json",
        {
            "status": "Development references only; no final-test or Tacit superiority claim.",
            "cases": len(cases),
            "decisions_per_model": len(expected),
            "tracks": metrics,
            "successful_api_cost_usd": str(cost),
            "budget": reports["jev"]["cumulative_budget"],
            "limitations": [
                "Tacit selects checkpoints on this development partition; it is not a final test.",
                "One general Laya checkpoint across all tracks; workflow specialist kept separate.",
                "Reference validation predictions are diagnostic, not training targets.",
                "Paired NLI input layouts share examples and are not independent evidence.",
                "Local evaluation ran during training; no latency comparison is valid here.",
                "Jev exposes rounded probabilities and zeros; NLL depends on this exposed output.",
                "Released references may have seen locally held-out examples.",
                "The fresh final gate remains unscored; dataset-specific license limits apply.",
            ],
        },
    )
    write(
        destination / "checksums.json",
        {p.name: sha256(p) for p in sorted(destination.iterdir()) if p.name != "checksums.json"},
    )
    print("Exported", len(cases), "validation cases and", len(expected), "decisions per model.")


if __name__ == "__main__":
    main()
