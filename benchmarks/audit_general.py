"""Audit corpus partitions, label counts and released Laya input truncation."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from general_data import load_corpus
from transformers import AutoTokenizer


def main():
    root = Path(".cache/general-v1")
    parts = {k: load_corpus(root, k) for k in ["train", "validation", "calibration", "test"]}
    ids = {k: {r["id"] for r in rows} for k, rows in parts.items()}
    premises = {
        k: {" ".join(r["state"].split()) for r in rows if r["id"].startswith("mnli/")}
        for k, rows in parts.items()
    }
    for i, a in enumerate(parts):
        assert len(ids[a]) == len(parts[a]), "duplicate IDs"
        for b in list(parts)[i + 1 :]:
            assert not ids[a] & ids[b], "split ID leakage"
            assert not premises[a] & premises[b], "NLI premise leakage"
    sys.path.insert(0, str(Path(".cache/laya-reference").resolve()))
    from laya.common import render_options, serialize_state

    tokenizer = AutoTokenizer.from_pretrained(".cache/laya-general/tokenizer")
    config = json.loads(Path(".cache/laya-general/rl_agent_config.json").read_text())
    counts = {"nli": Counter(), "typed": Counter()}
    details = []
    for row in parts["test"]:
        family = "nli" if row["id"].startswith("mnli/") else "typed"
        for name, q in row["questions"].items():
            internal = {"t": q["type"], "ins": q["instructions"], "crit": q.get("criteria")}
            opts = render_options(internal)
            option_lengths = [
                1
                + len(
                    tokenizer(" " + o.replace(tokenizer.mask_token, " "), add_special_tokens=False)[
                        "input_ids"
                    ]
                )
                for o in opts
            ]
            used_options = [min(n, 49) for n in option_lengths]
            if config["head_max_len"] - sum(used_options) < 16:
                per = max(4, (config["head_max_len"] - 16) // len(used_options))
                used_options = [min(n, per) for n in used_options]
            head_length = len(
                tokenizer(
                    q["type"]
                    + " question: "
                    + q["instructions"].replace(tokenizer.mask_token, " "),
                    add_special_tokens=False,
                )["input_ids"]
            )
            used_head = min(head_length, max(8, config["head_max_len"] - sum(used_options)))
            state_length = len(
                tokenizer(
                    serialize_state(row["state"]).replace(tokenizer.mask_token, " "),
                    add_special_tokens=False,
                )["input_ids"]
            )
            state_room = max(0, config["max_len"] - used_head - sum(used_options) - 4)
            truncated = {
                "instructions": head_length > used_head,
                "options": option_lengths != used_options,
                "state": state_length > state_room,
            }
            counts[family]["decisions"] += 1
            for k, v in truncated.items():
                counts[family][k + "_truncated"] += int(v)
            if any(truncated.values()):
                details.append({"id": row["id"], "question": name, **truncated})
    report = {
        "split_counts": {k: len(v) for k, v in parts.items()},
        "no_id_or_nli_premise_overlap": True,
        "file_sha256": {
            k: hashlib.sha256((root / (k + ".jsonl")).read_bytes()).hexdigest() for k in parts
        },
        "nli_test_relation_counts": dict(
            Counter(
                r["gold"]["relation"]["label"] for r in parts["test"] if r["id"].startswith("mnli/")
            )
        ),
        "nli_test_genres": dict(
            Counter(r["workflow"] for r in parts["test"] if r["id"].startswith("mnli/"))
        ),
        "laya_input_truncation": counts,
        "truncated_decisions": details,
        "notes": (
            "Laya shipped 512 total/192 head-token limits, 48-token option text limit. "
            "Tacit rejects oversize inputs; every case completed. "
            "MultiNLI corpus, not the authors' XNLI benchmark."
        ),
    }
    path = Path("results/general/data-audit.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print({k: v for k, v in report.items() if k != "truncated_decisions"})


if __name__ == "__main__":
    main()
