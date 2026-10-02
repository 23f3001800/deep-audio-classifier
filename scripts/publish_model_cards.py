"""Publish only reviewed README files; stop if the Hub revision changed."""
import argparse
import json
from pathlib import Path


def main():
    from huggingface_hub import HfApi, ModelCard, hf_hub_download

    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Upload the five README files")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "model-cards"
    manifest = json.loads((root / "publish-manifest.json").read_text())
    api = HfApi()
    if args.apply and api.whoami()["name"].lower() != "vikas25s":
        raise RuntimeError("Expected the Vikas25S account")
    pending = []
    for item in manifest["models"]:
        if not item["repo_id"].startswith("Vikas25S/"):
            raise ValueError("Unexpected model owner")
        path = (root / item["path"]).resolve()
        if root.resolve() not in path.parents or path.name != "README.md":
            raise ValueError("Only model-card README files can be uploaded")
        ModelCard.load(str(path)).validate()
        current = api.model_info(item["repo_id"]).sha
        if current != item["parent_commit"]:
            raise RuntimeError(f"{item['repo_id']} changed; review its current README before uploading")
        pending.append((item, path))
        print("Ready:", item["repo_id"])
    if not args.apply:
        print("Dry run only. Use --apply from an authenticated Vikas25S session to publish.")
        return
    receipts = []
    try:
        for item, path in pending:
            commit = api.upload_file(path_or_fileobj=str(path), path_in_repo="README.md",
                                     repo_id=item["repo_id"], repo_type="model",
                                     parent_commit=item["parent_commit"],
                                     commit_message="Document checkpoint usage, provenance and evaluation limits")
            downloaded = Path(hf_hub_download(item["repo_id"], "README.md", revision=commit.oid))
            if downloaded.read_bytes() != path.read_bytes():
                raise RuntimeError("Published README verification failed")
            receipts.append({"repo_id": item["repo_id"], "commit": commit.oid, "url": commit.commit_url})
            print("Published:", commit.commit_url)
    finally:
        (root / "publish-receipt.json").write_text(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()
