"""Fetch pinned NLI data, released Laya weights and optional research tasks."""

import argparse

from general_data import MNLI_REVISION
from huggingface_hub import snapshot_download

LAYA_GENERAL_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auxiliary", action="store_true")
    args = parser.parse_args()
    snapshot_download(
        "nyu-mll/multi_nli",
        repo_type="dataset",
        revision=MNLI_REVISION,
        local_dir=".cache/multi-nli",
        allow_patterns=["README.md", "data/*.parquet"],
    )
    snapshot_download(
        "convaiinnovations/laya",
        revision=LAYA_GENERAL_REVISION,
        local_dir=".cache/laya-general",
        allow_patterns=[
            "README.md",
            "model.safetensors",
            "encoder/config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "rl_agent_config.json",
        ],
    )
    if args.auxiliary:
        from general_v2_data import REVISIONS

        for repo, directory, pattern in [
            ("fancyzhx/ag_news", ".cache/ag-news", "data/*.parquet"),
            ("dair-ai/emotion", ".cache/emotion", "split/*.parquet"),
        ]:
            snapshot_download(
                repo,
                repo_type="dataset",
                revision=REVISIONS[repo],
                local_dir=directory,
                allow_patterns=["README.md", pattern],
            )


if __name__ == "__main__":
    main()
