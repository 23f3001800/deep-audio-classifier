"""Read public model metadata. Never print or persist authentication tokens."""
import json
import os
import urllib.request
from pathlib import Path
from urllib.error import HTTPError

root = Path("hub-inventory")
root.mkdir(exist_ok=True)
def read(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")

models = json.loads(read("https://huggingface.co/api/models?author=Vikas25S&full=true"))
inventory = {"models": [], "upload_credential_available": bool(os.getenv("HF_TOKEN"))}
for entry in models:
    model_id = entry["id"]
    item = {"id": model_id, "sha": entry.get("sha"), "pipeline_tag": entry.get("pipeline_tag"), "files": [s["rfilename"] for s in entry.get("siblings", [])]}
    for filename in ("README.md", "config.json", "preprocessor_config.json", "adapter_config.json"):
        try:
            item[filename] = read(f"https://huggingface.co/{model_id}/resolve/{entry.get('sha') or 'main'}/{filename}")
        except HTTPError as error:
            if error.code == 404:
                item[filename] = None
            else:
                raise
    inventory["models"].append(item)
(root / "inventory.json").write_text(json.dumps(inventory, indent=2))
print(json.dumps(inventory, indent=2))
