# Model card publication

Five README cards were validated against Hugging Face's card validator on 2 October 2026. All five cards were subsequently published and their downloaded contents verified. Exact commit URLs are recorded in `publish-receipt.json`.

[Validation run](https://github.com/23f3001800/deep-audio-classifier/actions/runs/37022289937).

For a future publication after refreshing the expected revisions, add a fine-grained Hugging Face token with write access to the five models listed in `publish-manifest.json` as the `HF_TOKEN` Actions secret in this GitHub repository. Re-run the **Validate and publish model cards** workflow run. It checks the Vikas25S account and the expected revision, updates only each model's README, and verifies the uploaded content. It does not change weights, configuration or model visibility.

Alternatively, in an authenticated local checkout of this branch:

```bash
python -m pip install huggingface_hub
hf auth login
python scripts/publish_model_cards.py
python scripts/publish_model_cards.py --apply
```

Do not paste the token into a chat or commit it to the repository. If a model has changed since inventory, the publisher stops so its current card can be reviewed before replacement.

`audio-smoke.json` records the successful fresh AST-v2 CPU inference test. This is not an independent genre-accuracy score. See `RECOMMENDATIONS.md` for the next models worth packaging.
