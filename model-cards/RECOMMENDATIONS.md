# Next models to publish

Start with useful, reproducible artifacts from work already completed. These are recommendations, not claims that new weights have been trained or uploaded.

## 1. Comment classifier and its full preprocessing pipeline

The [Comment Category Prediction project](https://github.com/23f3001800/Comment-Category-Prediction-Challenge) already describes a 70/30 LightGBM and Logistic Regression probability blend with dual TF-IDF and numeric features. Package the fitted vectorizers, numeric transforms, both classifiers, feature schema and a minimal prediction example together. Publishing just the classifier would leave users unable to reproduce its inputs.

The README's 0.8337 macro F1 is historical validation, and the blend weights were selected on that validation set. Use a fresh untouched split for a new score. Add a text-only baseline: internal platform features such as `if_2` may be unavailable to real users and deserve a leakage/availability audit. Report a group split by discussion thread and the rare class's precision and recall. This would add a distinct, practical NLP artifact to a Hub profile currently dominated by related AST checkpoints.

## 2. A smaller audio baseline

If the CRNN or other baseline checkpoint from the audio notebooks is still available, publish it with the exact preprocessing and label mapping. Evaluate it and AST-v2 on the same song-disjoint labelled set; report macro F1, size and CPU latency. If no trained checkpoint survives, this is a new training task, not a documentation-only upload.

## 3. Protein sequence tagger, once checkpoint evidence is recovered

The [Protein Secondary Structure Prediction repository](https://github.com/23f3001800/Protein-Secondary-Structure-Prediction) currently reads mainly as a competition brief. Publish the trained BiRNN/BiLSTM/BiGRU only after identifying the real checkpoint and its run. Include amino-acid vocabulary, Q3/Q8 label maps, sequence masking and inference examples. Validate on a split that controls sequence similarity, and keep token-level scores separate from any competition aggregate. Avoid a headline accuracy claim until the evidence is available.

## Improve the existing email adapter before adding another LLM

Recover its training-data provenance, verify loading on the recorded 4-bit base, and compare base versus adapter on unseen drafts. Score factual preservation separately from formatting. A documented, measured adapter is stronger portfolio evidence than another unbenchmarked fine-tune.

## Fix the current Hub packages first

- `messy-mashup-ast-v2`: keep as the clearest documented audio entry; independent genre evaluation remains pending.
- `ast-messy-mashup-classifier`: recover the original label mapping before changing configuration.
- `messy-mashup-ast`: recover its feature-extractor settings before adding a preprocessing file.
- `messy-mashup-ensemble`: verify the member list and loader; only one member checkpoint was visible in the inspected tree.
- `llama-3.2-email-formatter`: document adapter loading and resolve the previous card's Apache-2.0 statement against upstream Llama 3.2 terms.

Do not create duplicate model repositories solely to increase the model count.
