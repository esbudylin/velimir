# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#     "huggingface_hub>=0.27",
# ]
# ///
import argparse

from huggingface_hub import HfApi, get_token, login

from velimir.settings import DATASET_FILES, DATASETS_DIRECTORY

REPO_ID = "esbudylin/velimir"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload local datasets to the Hugging Face dataset repository."
    )
    parser.add_argument(
        "--repo-id", default=REPO_ID, help="Hugging Face dataset repository id"
    )
    parser.add_argument("--revision", default="main", help="Branch to update")
    parser.add_argument("--message", default="Update datasets", help="Commit message")
    args = parser.parse_args()

    if get_token() is None:
        login()

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)

    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        folder_path=DATASETS_DIRECTORY,
        allow_patterns=DATASET_FILES,
        commit_message=args.message,
    )

    print(f"Uploaded {', '.join(DATASET_FILES)} to {args.repo_id}")


if __name__ == "__main__":
    main()
