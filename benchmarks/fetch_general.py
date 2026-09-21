"""Fetch pinned human-labeled NLI data and the released general Laya weights."""

from general_data import MNLI_REVISION
from huggingface_hub import snapshot_download

LAYA_GENERAL_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"


def main():
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


if __name__ == "__main__":
    main()
