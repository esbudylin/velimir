# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#     "huggingface_hub>=0.27",
# ]
# ///
import argparse

from huggingface_hub import snapshot_download

from velimir.settings import DATASET_FILES, DATASETS_DIRECTORY

REPO_ID = "esbudylin/velimir"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download datasets from the Hugging Face dataset repository."
    )
    parser.add_argument(
        "--repo-id", default=REPO_ID, help="Hugging Face dataset repository id"
    )
    parser.add_argument("--revision", default="main", help="Revision to download")
    args = parser.parse_args()

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=DATASETS_DIRECTORY,
        allow_patterns=DATASET_FILES,
    )

    print(f"Downloaded {', '.join(DATASET_FILES)} from {args.repo_id}")


if __name__ == "__main__":
    main()
