# Self-Consistency as an Unsupervised Post-Training Signal for VQA

CS 289 graduate project — Matthew Bernard, Shiqi Yang — Fall 2026

## Goal

Can self-consistency serve as a reliable unsupervised post-training signal
for a vision-language model on visual question answering? We post-train
SmolVLM-500M on its own high-consensus outputs (no ground-truth labels
used for the training signal) and measure the effect on ScienceQA accuracy.

## Approach

1. **Baseline:** zero-shot SmolVLM-500M on ScienceQA (greedy + majority-vote).
2. **Consistency scoring:** for each question, sample `k=8` outputs at
   temperature `T>0`; also compute the softmax over answer options in a
   single forward pass. Compare the two consistency measures.
3. **Filtering:** retain questions whose mode frequency (or softmax score)
   exceeds threshold `τ`. Vary `τ ∈ {0, 0.5, 0.75, 1.0}`.
4. **Post-training:** LoRA fine-tune on the filtered pseudo-labels.
5. **Evaluation:** accuracy on held-out test set. Ablations over `k`, `T`,
   `τ`, LoRA rank, training epochs. Error-propagation study across
   multiple post-training rounds.

## Data

- **ScienceQA** (via HuggingFace Datasets). Main experiments on the
  image+text subset; text-only as secondary analysis.
- **VQAv2** — optional stretch goal.

## Proposal files

- `grad-project-proposal.qmd` — source
- `grad-project-proposal.pdf` — rendered
- `references.bib` — bibliography

## Setup

    conda create -n cs289 python=3.11
    conda activate cs289
    pip install -r requirements.txt

## Pipeline

    scripts/00_download_data.py      # fetch + cache ScienceQA
    scripts/01_baseline_eval.py      # zero-shot baseline
    scripts/02_self_consistency.py   # k-sampling + softmax consistency
    scripts/03_posttrain_lora.py     # (TBD) LoRA fine-tuning
    scripts/04_ablate.py             # (TBD) ablation grid

## Compute

SCF (Statistical Computing Facility) — primarily CPU cluster; GPU
allocation TBD. Fallbacks: Savio, Colab Pro. Baseline inference (k=8
sampling over ScienceQA) is embarrassingly parallel and runs on CPUs.

## Structure

    data/         # cached datasets (not committed)
    scripts/      # pipeline scripts
    notebooks/    # exploratory analysis
    configs/      # YAML configs for experiments

## Meetings

Weekly check-in — TBD.
