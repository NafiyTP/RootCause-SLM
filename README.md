# RootCause-SLM

A small pipeline for HDFS logs in two steps: first find the anomalous blocks, then explain
them with a small language model that runs locally.

- **Detection**: the raw lines are grouped by block id and each block is classified from
  its template counts (logistic regression, with PCA and a simple rule as comparison).
- **Explanation**: I fine-tuned Qwen2.5-1.5B-Instruct with LoRA so that, given a log line
  and its label, it returns a short cause and a 3-step reasoning as JSON.

The idea was to see how far a small local model can get, since sending infrastructure logs
to an external API is often not an option. The explanation model was trained on annotations
written by a larger model (Llama 3.3-70B through Groq), so it is a teacher-student setup:
the 70B model writes the explanations once, the 1.5B model learns to reproduce them.

The first version of this project only had the explanation part, with the label given as
input. That was the main weakness (the label has to come from somewhere), so I added the
detection step, a script that chains both, and a benchmark of speed and cost.

## Data

Logs and labels come from [Loghub](https://github.com/logpai/loghub) (HDFS_v1, a Yahoo
cluster, 2008, about 11M lines). Labels are per block in `anomaly_label.csv`. The raw log is
too big for the repo, `src/detect.py` reads it from wherever you downloaded it.

For the explanation model I give each line the label of its block. Llama 3.3-70B only writes
the `cause` and `raisonnement` fields, it never decides the label. The annotations are in
French, which is why the prompts in the code are in French too.

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

## Step 1: detection

A block is a session: all the lines that mention its id. Each line is reduced to a template
by masking block ids, IPs, paths and numbers (the same function the explanation baselines
use), and a block becomes a vector of template counts. Templates never seen in training go
into one `<UNK>` column instead of being dropped, because a new kind of message is
exactly the kind of thing you want to notice.

The split is chronological: blocks are sorted by their first line, the first 80% are used
for training and the last 20% for testing. A random split would let the model train on
blocks that happen after the test ones.

Three detectors:

- **WARN/ERROR rule**: a block is anomalous if one of its lines is not INFO. This is what you
  would do without any ML.
- **PCA** (unsupervised, the classic method on this dataset, Xu et al. 2009): fit PCA on
  normalized counts, keep 95% of the variance, and flag the blocks with a large
  reconstruction error. The threshold is the 97th percentile of the training scores, so the
  only thing it assumes is a rough anomaly rate.
- **Logistic regression** on log(1 + counts), `class_weight="balanced"` because anomalies
  are about 3% of the blocks.

Results on the first 100k lines of HDFS_v1 (7,940 blocks, test = last 1,588 blocks, 5.7%
anomalies):

| Detector | Precision | Recall | F1 |
|---|---:|---:|---:|
| WARN/ERROR rule | 1.000 | 0.011 | 0.022 |
| PCA (unsupervised) | 0.980 | 0.549 | 0.704 |
| Logistic regression | 0.982 | 0.593 | 0.740 |

The rule is almost useless: nearly every line of an anomalous block is an INFO line, so the
log level says nothing. Both models are very precise but miss about 40% of the anomalies here.
I looked at the missed blocks and they contain exactly the same events as normal blocks. My
explanation is that on a 100k-line extract many blocks are cut before the lines that make them
anomalous are written. If that is right, the detector needs complete sessions, which is a real
constraint for a streaming setup.

Detection runs at about 100k lines/s on a laptop CPU, parsing included. The detector is
saved in `modele_detection/detector.pkl` and the full numbers are in
[`results/detection/`](results/detection/).

## Step 2: explanation

### Training

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

### Evaluation

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

## Putting both together

`src/pipeline.py` takes a raw log file, scores every block with the logistic regression, and
for each flagged block picks the line that weighs the most in the decision: a template never
seen in training if there is one, otherwise the template with the largest coefficient. That
line goes to the fine-tuned model with the label "Anomaly". The output is a JSONL report with
the block, its score, the chosen line and the explanation.

The explanation is still about one line and not the whole block, because that is how the model
was trained. When I tested the pipeline on the last 20k lines of the extract, it flagged 1,105
blocks out of 2,624, almost all of them blocks with only one or two lines whose beginning was
cut off. On a live stream you would have to wait for a block to be closed before scoring it.

## Speed and cost

`src/benchmark.py` measures the fine-tuned model on a T4 (fp16, LoRA merged into the weights)
with one request at a time and with batching, then compares the cost per 1,000 explanations
with calling Llama 3.3-70B on Groq ($0.59 / $0.79 per million input / output tokens). The
teacher cost is estimated from the token counts of the same prompts and answers.

| Setup | s / line | lines / s | USD per 1k lines |
|---|---:|---:|---:|
| Qwen2.5-1.5B + LoRA, batch 1 | TODO | TODO | TODO |
| Qwen2.5-1.5B + LoRA, batch 16 | TODO | TODO | TODO |
| Llama 3.3-70B on Groq (estimate) | | | TODO |

## Limitations

- **The explanation model justifies a label, it does not find a cause.** Labels are per block
  and the model sees one line. All 250 test anomalies are INFO lines whose templates also
  appear with the Normal label, so there is nothing in the line itself that explains the
  anomaly. The teacher often makes up a cause, and about a quarter of its explanations use the
  label itself as the evidence.
- **Small and repetitive training data.** 69 training anomalies over 29 different causes, and
  only 15 message templates overall. The validation perplexity is already 1.19 after one epoch,
  and on the test set the model only uses 8 distinct causes (the references have 19).
- **The metrics compare to the teacher, not to the truth.** There are no human-validated root
  causes for this dataset.
- **The detector needs complete sessions and labels.** The logistic regression is supervised,
  and both models fail on blocks that are cut. PCA works without labels but is a bit weaker.
- **Only HDFS.** I did not test on another system (BGL or Thunderbird from Loghub), where
  templates are much more varied.

What I would do next: give the model the whole flagged block instead of one line, and
re-annotate at the block level so that the cause can actually be read from the input.

## Repo structure

```
data/                  train and test sets for the explanation model (JSON and CSV)
modele_hdfs/           LoRA adapter, tokenizer, training history
modele_detection/      trained detector (template list + logistic regression)
results/detection/     detection metrics
results/eval_n527/     explanation metrics, predictions and examples on the full test set
results/benchmark/     speed and cost of the fine-tuned model
src/detect.py          block-level detection (rule, PCA, logistic regression)
src/dataset.py         Dataset and collator (ChatML, loss masking, dynamic padding)
src/train.py           LoRA training with the weighted sampler
src/evaluate.py        baselines, zero-shot, few-shot and fine-tuned evaluation
src/inference.py       explain one log line with the fine-tuned model
src/pipeline.py        detect anomalous blocks in a log file, then explain them
src/benchmark.py       latency, throughput and cost
evaluation_colab.ipynb runs the evaluation on a Colab T4
```

## Usage

```bash
pip install -r requirements.txt

# detection (CPU), needs HDFS.log and anomaly_label.csv from Loghub HDFS_v1
python src/detect.py --log path/to/HDFS.log --labels path/to/anomaly_label.csv

# explanation model
python src/train.py                      # training (GPU)
python src/evaluate.py                   # full evaluation (GPU, about 30 min on a T4)
python src/evaluate.py --no-llm          # baselines only, CPU
python src/inference.py --log "081109 203615 148 WARN dfs.DataNode\$DataXceiver: Got exception while serving blk_..." --label Anomaly

# both steps together
python src/pipeline.py --log some_logs.log --top 10
python src/pipeline.py --log some_logs.log --no-llm   # detection only

# speed and cost (GPU)
python src/benchmark.py
```
