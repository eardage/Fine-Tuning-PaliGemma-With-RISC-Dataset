# DI725 Assignment 3 — PaliGemma QLoRA Fine-Tuning on RISC

Fine-tuning PaliGemma-3B on the RISC remote-sensing image captioning dataset
using QLoRA, for METU DI725.

## Results
- Baseline BLEU-4 (n=100): 0.000
- Fine-tuned BLEU-4 (n=100): 0.318

## Files
- `notebook.ipynb` — full training and evaluation pipeline
- `report.pdf` — 1-page report
- `adapter/` — trained LoRA adapter weights
- `results.json` — predictions and BLEU scores
- `requirements.txt` — pinned package versions

## Reproduction
Run `notebook.ipynb` on Kaggle with GPU T4/P100, using a HuggingFace token 
with PaliGemma access in the `HF_TOKEN` secret.
