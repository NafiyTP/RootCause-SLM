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
detection step, a script that chains both, a benchmark of speed and cost, and an audit that
checks the explanations against the raw logs instead of against the teacher.

Main results:

- Detection on the full HDFS_v1 log (11M lines, chronological split): logistic regression
  reaches F1 0.973 (precision 0.955, recall 0.992). The log level alone finds a quarter of
  the anomalies. On BGL, a harder system, the same approach drops to F1 0.676 and a simple
  log-level rule does better (0.846), because most test windows contain templates never seen
  in training.
- The fine-tuned 1.5B model reproduces the teacher better than every baseline (ROUGE-L
  0.506 on the reasoning against 0.456 for template retrieval), and with batching it is
  about 7 times cheaper than calling the 70B teacher.
- But neither the teacher nor the student finds the real cause: on 250 anomalous test
  lines, 0 explanations from the student and 1 from the teacher mention what is actually
  wrong with the block. What is wrong is almost always somewhere else in the block (a
  write that never finished, a failed delete), not in the line the model sees. A simple
  "which usual events are missing" check in the pipeline is more useful than the LLM here.

## Data

Logs and labels come from [Loghub](https://github.com/logpai/loghub) (HDFS_v1, Hadoop jobs on more
than 200 Amazon EC2 nodes, 2008, about 11M lines). Labels are per block in `anomaly_label.csv`. The raw log is
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

Results on the full HDFS_v1 log (11,175,629 lines, 575,061 blocks, 54 templates; test =
last 115,013 blocks, 1.5% anomalies):

| Detector | Precision | Recall | F1 | Flagged |
|---|---:|---:|---:|---:|
| WARN/ERROR rule | 1.000 | 0.248 | 0.398 | 417 |
| PCA (unsupervised) | 0.379 | 0.051 | 0.090 | 227 |
| Logistic regression | 0.955 | 0.992 | 0.973 | 1,744 |

- **The rule** never raises a false alarm, but three quarters of the anomalous blocks only
  contain INFO lines.
- **PCA** ranks the blocks well (ROC AUC 0.98 on the test set) but its threshold does not
  transfer: it is fixed on the training period, where anomalies are 3.3% of the blocks,
  against 1.5% in the test period, and the scores shift between the two. Setting the
  threshold well needs either labels or a known anomaly rate.
- **Logistic regression** finds almost every anomaly. The templates with the largest weights
  are the ones you would expect: a redundant `addStoredBlock`, a block that belongs to no
  file, a failed delete, a replication timeout.

The detector needs complete sessions. On an extract of the first 105k lines, the same model
only gets F1 0.27: the most recent test blocks are cut at the end of the extract, look like
writes that never finished, and get flagged. On a live stream you would have to wait for a
block to be closed before scoring it.

Detection runs at about 115k lines/s on one CPU core, parsing included (about 2 minutes
for the full log). The detector is saved in `modele_detection/detector.pkl` as plain lists
(templates, weights, and how often each template appears in normal blocks), so it loads
with any scikit-learn version. The full numbers are in
[`results/detection/`](results/detection/).

### Does it carry over? BGL

HDFS is an easy dataset: 54 templates and very regular sessions. `src/detect_bgl.py` runs the
same detectors on BGL (Blue Gene/L supercomputer, Loghub, 4.7M lines). There is no session id,
so a session is a 60-minute window over the whole machine, anomalous if it contains at least
one alert line. The level rule flags a window if one of its lines is FATAL, FAILURE, SEVERE or
ERROR. Chronological split, test = last 724 windows (18.6% anomalous):

| Detector | Precision | Recall | F1 | Flagged |
|---|---:|---:|---:|---:|
| Level rule (FATAL/FAILURE/SEVERE/ERROR) | 0.734 | 1.000 | 0.846 | 184 |
| PCA (unsupervised) | 0.203 | 0.541 | 0.295 | 359 |
| Logistic regression | 0.972 | 0.518 | 0.676 | 72 |

The ranking is the opposite of HDFS: the simple level rule wins. The logistic regression is
still very precise, but it misses half of the anomalous windows. The reason shows in the
templates: my masking (numbers, hex values, IPs, paths) was written for HDFS, and on BGL it
leaves 13,603 templates in training, and 412 of the 724 test windows (57%) contain a
template never seen before. All of those land in the single `<UNK>` column, so a new kind of
failure looks almost like a new kind of normal message. Detection on counts of templates
only works if the templates are stable; on a real system that needs a proper log parser
(for example Drain) and a model that can handle new templates, which I did not do here.
Full numbers in [`results/detection_bgl/`](results/detection_bgl/).

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

### Audit: do the explanations find the real cause?

The scores above only say how close each system gets to the teacher. To check the
explanations against the logs themselves, `src/audit_explanations.py` rebuilds the whole
block of each anomalous test line from `HDFS.log` and looks for the block-level evidence:
the events that 90% of normal blocks have and this one does not, or rare events it contains.
Then it checks whether the cause and reasoning mention that evidence. I also read 50 of them
by hand (25 anomalies, 25 normal lines) to make sure the keyword check misses nothing.

| What is wrong with the block | Blocks |
|---|---:|
| the write never completed (no `addStoredBlock`, no `PacketResponder terminating`) | 101 (40%) |
| a failed delete (`BlockInfo not found in volumeMap`) | 82 (33%) |
| `addStoredBlock` for a block that belongs to no file | 47 (19%) |
| extra replication | 16 (6%) |
| empty packet | 4 (2%) |

| System | Mentions this evidence |
|---|---:|
| Llama 3.3-70B (teacher) | 1 / 250 |
| Qwen2.5-1.5B + LoRA | 0 / 250 |

Both models tell one of two stories depending on the line they get: an `allocateBlock` line
becomes "Erreur d'allocation de bloc", a `Receiving block` line becomes "Connexion non
autorisée". Neither is supported by the logs: the allocation itself succeeded, and nothing
in the block points to an unauthorized connection. On normal lines the explanations are
fine, because they only describe the event. So the fine-tuning worked (the student copies
the teacher well), but what it copies is a guess, because the cause is not in the input.
Details and the hand-checked sample are in [`results/audit/`](results/audit/).

## Putting both together

`src/pipeline.py` takes a raw log file, scores every block with the logistic regression, and
for each flagged block:

- lists the **missing events**: templates that at least 90% of normal training blocks
  contain but this block does not
- picks the line that weighs the most in the decision (a template never seen in training if
  there is one, otherwise the one with the largest weight) and sends it to the fine-tuned
  model with the label "Anomaly"

The output is a JSONL report with the block, its score, the missing events, the chosen line
and the explanation.

I tested it on 3,000 consecutive blocks from the test period, with all their lines (38,957
lines, 15 anomalous blocks). It flagged exactly the 15 anomalies, with no false alarm, in
0.4 s. For 12 of them the missing events say what happened: the block was allocated and
the transfer started, but it was never stored (`addStoredBlock` and `PacketResponder
terminating` never appear). For those 12, the line sent to the language model is the
`allocateBlock` line, and from that line alone it can only guess. For the other 3, the chosen
line is the anomalous event itself (a redundant `addStoredBlock`, an empty packet), so the
line selection does its job when the anomaly is something that happened rather than
something that did not. The missing-events list is the better explanation for the first
kind, and it costs nothing.

## Speed and cost

`src/benchmark.py` measures the fine-tuned model on a T4 (fp16, LoRA merged into the weights)
with one request at a time and with batching, then compares the cost per 1,000 explanations
with calling Llama 3.3-70B on Groq ($0.59 / $0.79 per million input / output tokens). The
teacher cost is estimated from the token counts of the same prompts and answers.

Measured on 64 test lines (average request: 161 tokens in, 92 tokens out), with a T4 at
$0.35 per hour:

| Setup | s / line | lines / s | tokens / s | USD per 1k lines |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B + LoRA, batch 1 | 2.918 | 0.34 | 30 | 0.284 |
| Qwen2.5-1.5B + LoRA, batch 16 | 0.251 | 3.99 | 344 | 0.024 |
| Llama 3.3-70B on Groq (estimate) | | | | 0.168 |

One request at a time, the small model is slow (about 3 s per explanation) and actually
costs more than the API, because the GPU is paid for while it waits. With batches of 16 it
is about 12 times faster and about 7 times cheaper than the teacher, and it fits in 3.4 GB
of GPU memory. So running it locally only pays off when explanations can be grouped, which
is the case when the detector flags many blocks at once. The GPU price is the GPU alone, not
the whole VM, so the real gap is a bit smaller. Full numbers in
[`results/benchmark/`](results/benchmark/).

## Limitations

- **The explanation model justifies a label, it does not find a cause.** Labels are per block
  and the model sees one line. All 250 test anomalies are INFO lines whose templates also
  appear with the Normal label, so there is nothing in the line itself that explains the
  anomaly. The audit confirms it: 0 of 250 student explanations mention the real evidence.
- **Small and repetitive training data.** 69 training anomalies over 29 different causes, and
  only 15 message templates overall. The validation perplexity is already 1.19 after one epoch,
  and on the test set the model only uses 8 distinct causes (the references have 19).
- **The metrics compare to the teacher, not to the truth.** There are no human-validated root
  causes for this dataset. The audit checks against the logs, but with keywords and a
  50-example manual check, not with expert labels.
- **The detector needs complete sessions and labels.** The logistic regression is supervised,
  and it fails on blocks that are cut. PCA works without labels but its threshold does not
  transfer from one period to the next.
- **The detector does not generalize as is.** On BGL the template masking breaks down (13,603
  templates, 57% of test windows with an unseen one) and the logistic regression loses to a
  log-level rule. It needs a real parser and a way to handle new templates.

What I would do next: give the model the whole flagged block (or the list of missing and
rare events) instead of one line, re-annotate at the block level so that the cause can
actually be read from the input, and replace the regex masking with a real parser (Drain) so
that detection holds on systems like BGL.

## Repo structure

```
data/                  train and test sets for the explanation model (JSON and CSV)
modele_hdfs/           LoRA adapter, tokenizer, training history
modele_detection/      trained detector (template list + logistic regression)
results/detection/     detection metrics (full HDFS_v1 log)
results/detection_bgl/ detection metrics on BGL
results/audit/         explanations checked against the raw blocks
results/eval_n527/     explanation metrics, predictions and examples on the full test set
results/benchmark/     speed and cost of the fine-tuned model
src/detect.py          block-level detection (rule, PCA, logistic regression)
src/detect_bgl.py      the same detectors on BGL, with time windows as sessions
src/audit_explanations.py  checks the explanations against the whole block
src/dataset.py         Dataset and collator (ChatML, loss masking, dynamic padding)
src/train.py           LoRA training with the weighted sampler
src/evaluate.py        baselines, zero-shot, few-shot and fine-tuned evaluation
src/inference.py       explain one log line with the fine-tuned model
src/pipeline.py        detect anomalous blocks in a log file, then explain them
src/benchmark.py       latency, throughput and cost
src/app.py             Gradio demo: detection + live explanations (GPU)
paper/                 write-up of the whole project (LaTeX source and PDF)
evaluation_colab.ipynb runs the evaluation on a Colab T4
pipeline_benchmark_colab.ipynb  runs the pipeline and the benchmark on a Colab T4
```

## Usage

```bash
pip install -r requirements.txt

# detection (CPU), needs HDFS.log and anomaly_label.csv from Loghub HDFS_v1
python src/detect.py --log path/to/HDFS.log --labels path/to/anomaly_label.csv
python src/detect_bgl.py --log path/to/BGL.log
python src/audit_explanations.py --log path/to/HDFS.log

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

# live demo (GPU, e.g. Colab; --share prints a public link)
pip install gradio "huggingface-hub<1.0"
python src/app.py --share
```
