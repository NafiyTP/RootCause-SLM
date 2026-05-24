# Rootcause-SLM: Local SLM for AIOps & Root Cause Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Custom%20Data%20Pipeline-EE4C2C?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green)

##  Abstract

In modern Cloud infrastructure, identifying the root cause of a system failure among millions of log lines is a critical challenge. While large proprietary LLMs (like GPT-4) can analyze logs, enterprise data privacy policies strictly prohibit sending sensitive server logs to external APIs. 

**Rootcause-SLM** solves this by fine-tuning a Small Language Model (SLM) — specifically Qwen-1.5B — to perform high-accuracy Root Cause Analysis (RCA) **100% locally**. Inspired by recent research in targeted SLM distillation, it doesn't just classify errors; it generates a deterministic, step-by-step **Chain-of-Thought (CoT)** reasoning before outputting a strict JSON format for downstream automation.

##  Key Engineering Features

- **Privacy-First & Cost-Efficient:** Runs locally with only 1.5 Billion parameters, requiring minimal VRAM compared to massive generic models.
- **Data Distillation Pipeline:** Synthetic CoT datasets generated using larger teacher models (Gemini 1.5 Flash) built on top of the open-source *Loghub* dataset.
- **Custom PyTorch Architecture:** Avoids high-level wrapper abstractions by implementing custom `torch.utils.data.Dataset` and `DataCollator` for highly optimized tensor padding and GPU memory management.
- **Deterministic Evaluation:** Strict structural constraints during Supervised Fine-Tuning (SFT) force the model to output valid JSON, allowing for rigorous `Pass@1` (Exact Match) automated evaluation without relying on "LLM-as-a-Judge" subjectivity.

##  Architecture & Pipeline

1. **Data Engineering (`data/`):** Raw logs are filtered, templates are extracted, and a teacher model generates the ground-truth reasoning steps.
2. **Low-Level Formatting (`src/dataset.py`):** Raw text and JSON structures are tokenized and packed into uniform matrices using a custom PyTorch collator, managing `-100` padding for optimal Loss calculation.
3. **Training (`src/train.py`):** The model undergoes SFT on high-end GPUs, actively penalizing formatting deviations to guarantee strict JSON compliance.

##  Quick Start
