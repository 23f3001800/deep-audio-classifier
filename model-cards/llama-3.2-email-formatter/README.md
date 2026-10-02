---
library_name: peft
pipeline_tag: text-generation
base_model: unsloth/llama-3.2-1b-instruct-unsloth-bnb-4bit
language:
- en
tags:
- llama
- lora
- email-formatting
---

# Llama 3.2 Email Formatter

This repository contains a LoRA adapter for an email-formatting experiment by Vikas. It contains adapter weights and a tokenizer, rather than a standalone copy of the base language model.

## Checkpoint details

- Inspected revision: `c33e0911c4446d7deff6f1d4ea127cc931c6c814`.
- Base checkpoint recorded in `adapter_config.json`: `unsloth/llama-3.2-1b-instruct-unsloth-bnb-4bit`.
- Adapter: LoRA, rank 16, alpha 16, dropout 0.05.
- Target modules: query, key, value and output projections, plus gate, up and down projections.
- Task: causal language modelling. The saved adapter config records PEFT 0.18.1.

## Intended use

Explore formatting an English email draft while preserving the supplied facts. Review every output before using it: a language model can invent a recipient, date, commitment or other detail that was absent from the input. This checkpoint does not send email.

## Loading and reproducibility

Load this repository as a PEFT adapter on its recorded base checkpoint. The base is a bitsandbytes 4-bit model, so the runtime must support that quantization setup. Use the repository's tokenizer and chat template. A verified end-to-end inference example is still pending; successful adapter upload alone does not establish that generation works in a fresh environment.

The training dataset, split, prompt format, seed, optimizer settings and training duration are not documented in the inspected repository. Do not infer those settings from the adapter configuration. No held-out formatting, factual-preservation or latency results have been published with this checkpoint.

## Evaluation needed

Compare the base model and this adapter on the same unseen email drafts. Measure format adherence and factual preservation separately, and record examples of added or removed facts. Include ambiguous requests and drafts containing instruction-like text. Report the checkpoint revisions, test prompts and decoding settings alongside results.

## License information

The previous generated card labelled this repository Apache-2.0. That label alone does not establish redistribution rights for the base model. The [upstream base card](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit) identifies Llama 3.2 terms. The adapter author's intended license and the training-data terms still need to be recorded; this card does not grant new rights or relicense the checkpoint.
