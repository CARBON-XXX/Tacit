"""Download the public, pinned artifacts required by the workflow experiments."""

import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download
from typed_common import DATA_REVISION


def main():
    sources = [
        (
            "LocalLLaMA/typed-decisions",
            "dataset",
            DATA_REVISION,
            ".cache/typed-data",
            ["README.md", "all/train-00000-of-00001.parquet", "all/test-00000-of-00001.parquet"],
        ),
        (
            "answerdotai/ModernBERT-base",
            "model",
            "8949b909ec900327062f0ebf497f51aef5e6f0c8",
            ".cache/modernbert-base",
            [
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "README.md",
            ],
        ),
        (
            "convaiinnovations/laya-typed-decisions",
            "model",
            "f9ab0b228f0fc0f14d873dbc99038f135c2da1b2",
            ".cache/laya-typed",
            None,
        ),
    ]
    for repo, kind, revision, destination, files in sources:
        snapshot_download(
            repo_id=repo,
            repo_type=kind,
            revision=revision,
            local_dir=destination,
            allow_patterns=files,
            token=False,
        )
    path = Path(".cache/laya-reference")
    revision = "6a5819129eb220570792e417e49723d697efd76f"
    if not path.exists():
        subprocess.run(
            ["git", "clone", "--no-checkout", "https://github.com/NandhaKishorM/laya", str(path)],
            check=True,
        )
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", revision], check=True)
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != revision:
        raise RuntimeError(
            "existing Laya source is at a different revision; use an isolated pinned checkout"
        )
    print("Pinned data, encoder, Laya weights and source are ready. Licenses remain unchanged.")


if __name__ == "__main__":
    main()
