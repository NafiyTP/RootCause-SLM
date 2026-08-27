# Rootcause-SLM

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Custom%20Data%20Pipeline-EE4C2C?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green)

Fine-tuning of Qwen2.5-1.5B for anomaly detection and root cause analysis on HDFS logs — fully local, no external API at inference time.

---

## Motivation

Sending server logs to GPT-4 or Claude raises an obvious problem in enterprise environments: logs contain sensitive infrastructure data. This project explores an alternative — a small model (1.5B parameters) fine-tuned on domain-specific data, running locally and producing structured output that can be consumed directly by downstream automation.

The approach is inspired by teacher-student distillation: a large model (Llama 3.3-70B via Groq) generates annotations, a small model learns from them. The result is a specialized, lightweight model that requires no internet access at inference time.

---

## Dataset

Source: [Loghub](https://github.com/logpai/loghub) — 2,000 real HDFS log lines collected from a Yahoo cluster in 2008, with `Normal`/`Anomaly` labels provided by [Loglizer](https://github.com/logpai/loglizer).

**Class imbalance:** 96.5% Normal / 3.5% Anomaly, addressed with `WeightedRandomSampler` to oversample anomalies during training.

**Annotation:** each log is enriched by Llama 3.3-70B with a `cause` and a 3-step `reasoning`. Ground truth labels always come from Loglizer — the LLM only generates the explanation text.

```json
{
  "log": "081109 203615 148 WARN dfs.DataNode: Got exception while serving blk_38865049...",
  "label": "Anomaly",
  "cause": "Network exception during block transfer between DataNodes",
  "reasoning": "Step 1: The DataNode was attempting to serve a block... Step 2: ..."
}
```

---

## Pipeline

```
hdfs_dataset.json
      │
      ▼
 dataset.py       Tokenization, ChatML formatting, label masking (-100)
      │
      ▼
  train.py        LoRA on Qwen2.5-1.5B, WeightedRandomSampler, cosine LR schedule
      │
      ▼
modele_hdfs/      Saved LoRA adapters
      │
      ▼
 inference.py     Log in → {cause, reasoning} JSON out
```

---

## Technical choices

**Qwen2.5-1.5B-Instruct** — Alibaba Cloud, Apache 2.0 license, trained on 18 trillion tokens. Chosen for its strong reasoning-to-size ratio and because it fits on a T4 GPU (16 GB).

**LoRA (r=16, alpha=32)** — instead of updating 1.5B parameters, we train two small matrices $A \in \mathbb{R}^{r \times d}$ and $B \in \mathbb{R}^{d \times r}$ such that $\Delta W = BA$. Only 4.36M parameters are trained (0.28% of total), on the `q_proj`, `k_proj`, `v_proj`, `o_proj` attention layers.

**Label masking** — cross-entropy is computed only on response tokens. Prompt tokens are set to `-100` (ignored by PyTorch). Without this, the model tries to predict its own context — wasted gradient.

**Dynamic padding** — the collator pads to the longest sequence in each batch, not to `max_length`. Saves VRAM at every training step.

---

## Repo structure

```
.
├── data/
│   ├── hdfs_dataset.json        # training set (1999 annotated logs)
│   └── hdfs_test_dataset.json   # held-out test set (527 logs)
├── src/
│   ├── dataset.py      # HDFSLogDataset + HDFSDataCollator
│   ├── train.py        # training loop + LoRA
│   ├── evaluate.py     # compares fine-tuned model vs baselines (ROUGE-L, BERTScore)
│   └── inference.py    # single-log inference with the fine-tuned model
├── modele_hdfs/         # LoRA adapter weights (generated after training)
├── results/             # evaluation output (resultats_evaluation.json)
├── requirements.txt
└── README.md
```

---

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Training — on Google Colab (T4 GPU recommended):

```python
!python src/train.py
```

Local (CPU, slow):

```bash
python src/train.py
```

Evaluation (compares fine-tuned model to the two baselines):

```bash
python src/evaluate.py          # full test set
python src/evaluate.py --n 50   # quick run on 50 examples
```

Inference on a single log:

```bash
python src/inference.py --log "081109 203615 148 WARN dfs.DataNode: Got exception while serving blk_38865049..."
```

---

## Results

### Training

| Epoch | Train loss | Val loss | Train PPL | Val PPL |
|-------|-----------|----------|-----------|---------|
| 1     | 0.419     | 0.176    | 1.52      | 1.19    |
| 2     | 0.094     | 0.123    | 1.10      | 1.13    |
| 3     | 0.060     | 0.117    | 1.06      | 1.12    |
| 4     | 0.050     | 0.100    | 1.05      | 1.10    |
| 5     | 0.049     | 0.099    | 1.05      | 1.10    |

Best checkpoint: epoch 5 (val_loss = 0.099). Full history in `modele_hdfs/historique.json`.

### Evaluation

Run `python src/evaluate.py` (add `--n 50` for a quick run) to compare the fine-tuned model
against the two baselines (most frequent cause, zero-shot Qwen). Results are written to
`results/resultats_evaluation.json`.

---

## Limitations

The dataset is small (2,000 examples) and not very diverse — HDFS logs are highly repetitive, which explains the very low perplexity from epoch 1. Generalization to other log systems (BGL, Thunderbird) has not been evaluated and would likely show lower performance, since the model has only seen HDFS-specific patterns.
