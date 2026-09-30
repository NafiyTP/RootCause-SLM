# RootCause-SLM

I fine-tuned Qwen2.5-1.5B-Instruct with LoRA to explain HDFS log lines: given a line and its
Normal/Anomaly label, it returns a short cause and a 3-step reasoning as JSON. The idea was to
see how far a small model running locally can get, since sending infrastructure logs to an
external API is often not an option.

The training annotations come from a larger model (Llama 3.3-70B through Groq), so this is a
teacher-student setup: the 70B model writes the explanations once, the 1.5B model learns to
reproduce them.

## Data

Logs and labels come from [Loghub](https://github.com/logpai/loghub) (HDFS_v1, a Yahoo cluster,
2008). Labels are per block in `anomaly_label.csv`, and I give each line the label of its block.
Llama 3.3-70B only writes the `cause` and `raisonnement` fields; it never decides the label.
The annotations are in French, which is why the prompts in the code are in French too.

| Split | Lines | Normal / Anomaly |
|---|---:|---|
| Train (`data/hdfs_dataset.json`), 85/15 train/val | 1,999 | 1,930 / 69 |
| Test (`data/hdfs_test_dataset.json`) | 527 | 277 / 250 |

The test set was built from blocks that never appear in the training set, with anomalies
oversampled so the metrics say something about them.

Example entry:

```json
{
  "log": "081109 203615 148 WARN dfs.DataNode$DataXceiver: Got exception while serving blk_38865049...",
  "label": "Anomaly",
  "cause": "...",
  "raisonnement": "Étape 1 : ... Étape 2 : ... Étape 3 : ..."
}
```

## Training

- Qwen2.5-1.5B-Instruct, LoRA r=16, alpha=32 on `q_proj`, `k_proj`, `v_proj`, `o_proj`
  (4.36M trainable parameters, 0.28% of the model)
- ChatML prompt, loss only on the answer tokens (prompt tokens set to -100)
- anomalies are 3.5% of the training lines, so a `WeightedRandomSampler` draws both classes
  about equally often
- AdamW, lr 2e-4, 5% warmup then cosine decay, effective batch size 16, 5 epochs on a Colab T4

| Epoch | Train loss | Val loss | Val perplexity |
|---:|---:|---:|---:|
| 1 | 0.419 | 0.176 | 1.19 |
| 2 | 0.094 | 0.123 | 1.13 |
| 3 | 0.060 | 0.117 | 1.12 |
| 4 | 0.050 | 0.100 | 1.10 |
| 5 | 0.049 | 0.099 | 1.10 |

The adapter from epoch 5 is in `modele_hdfs/`.

## Evaluation

I compare the fine-tuned model with three simple baselines and with the base model prompted
zero-shot and few-shot, on all 527 test lines. Decoding is greedy and the fine-tuned model gets
exactly the training prompt. The references are the Llama 3.3-70B annotations, so these scores
measure how close each system gets to the teacher, not whether the cause is actually right.

| System | Valid JSON | Cause exact match | ROUGE-L cause | ROUGE-L reasoning | BERTScore reasoning |
|---|---:|---:|---:|---:|---:|
| Most frequent cause | 100% | 0.461 | 0.594 | 0.347 | 0.797 |
| Majority cause per label | 100% | 0.461 | 0.589 | 0.356 | 0.783 |
| Template retrieval | 100% | 0.548 | 0.860 | 0.456 | 0.825 |
| Qwen2.5-1.5B zero-shot | 6.5% | 0.000 | 0.008 | 0.017 | 0.048 |
| Qwen2.5-1.5B few-shot (3 examples) | 100% | 0.000 | 0.393 | 0.292 | 0.778 |
| **Qwen2.5-1.5B + LoRA** | **100%** | **0.617** | **0.874** | **0.506** | **0.836** |

Template retrieval is the baseline that matters: for each test line it copies the annotation
of a training line with the same message template and label. The fine-tuned model beats it by
+0.050 ROUGE-L on the reasoning (paired bootstrap 95% CI [0.038, 0.064]) and doubles the exact
cause match on anomalies (0.284 vs 0.140). On the reasoning of anomalous lines, though, the two
are tied (0.455 vs 0.461).

The base model mostly fails on format in zero-shot: it wraps the JSON in code fences, breaks
strings over several lines or runs out of tokens. Three examples in the prompt fix the format,
but it then paraphrases the causes and never matches the reference wording.

Per-label numbers, confidence intervals, every prediction and side-by-side examples are in
[`results/eval_n527/`](results/eval_n527/).

Note: the first version of this README reported numbers from a 20-example check (all Normal
lines). A resume bug in the old evaluation script made the full run reuse those results.
The table above replaces them.

## Limitations

This is the part I would change first if I redid the project.

- **The labels are per block, the input is a single line.** A block is anomalous as a whole,
  and most of its lines are perfectly ordinary. All 250 test anomalies are INFO lines whose
  templates also appear with the Normal label. There is nothing in the line itself that explains
  the anomaly, so the teacher often makes up a cause, and about a quarter of its explanations
  use the label itself as the evidence. The model learns to justify a label, not to find a cause.
- **Small and repetitive data.** 69 training anomalies over 29 different causes, and only 15
  message templates overall. The validation perplexity is already 1.19 after one epoch, and on
  the test set the model only uses 8 distinct causes (the references have 19).
- **The metrics compare to the teacher, not to the truth.** There are no human-validated root
  causes for this dataset.
- **It is not a detector.** The label is an input. `src/inference.py` needs it from somewhere else.

A better design would group lines by `block_id` into sessions, detect anomalies at the session
level (DeepLog style, or a classifier on event counts), and only then ask the small model to
explain the flagged session with its full context.

## Repo structure

```
data/                  train and test sets (JSON and CSV)
modele_hdfs/           LoRA adapter, tokenizer, training history
results/eval_n527/     metrics, predictions and examples on the full test set
src/dataset.py         Dataset and collator (ChatML, loss masking, dynamic padding)
src/train.py           LoRA training with the weighted sampler
src/evaluate.py        baselines, zero-shot, few-shot and fine-tuned evaluation
src/inference.py       explain one log line with the fine-tuned model
evaluation_colab.ipynb runs the evaluation on a Colab T4
```

## Usage

```bash
pip install -r requirements.txt

python src/train.py                      # training (GPU)
python src/evaluate.py                   # full evaluation (GPU, about 30 min on a T4)
python src/evaluate.py --n 20            # quick check
python src/evaluate.py --no-llm          # baselines only, CPU
python src/inference.py --log "081109 203615 148 WARN dfs.DataNode\$DataXceiver: Got exception while serving blk_..." --label Anomaly
```
