# RootCause-SLM

Anomaly detection on HDFS logs, with an experiment on explaining the anomalies with a small
language model that runs locally.

**Live demo:** https://nafiytp.github.io/RootCause-SLM/ (runs in the browser: paste logs, edit a block,
move the detection threshold, and ask the small model (WebGPU) or the large one (your Groq key) why)

- **Detection** (the main part): the raw lines are grouped by block id and each block is
  classified from its template counts. A logistic regression is compared with a log-level
  rule and with PCA, on the full HDFS_v1 log and on a second system (BGL).
- **Explanation** (the experiment): I fine-tuned Qwen2.5-1.5B-Instruct with LoRA on
  explanations written by a large model, to see whether a small local model can replace the
  large one, since sending infrastructure logs to an external API is often not an option.
  v1 gave the model one line of the flagged block; v2 gives it a summary of the whole block.

Main results:

- **Detection works.** On the full HDFS_v1 log (11M lines, 575k blocks, chronological split)
  the logistic regression reaches F1 0.973 (precision 0.955, recall 0.992). The log level
  alone finds a quarter of the anomalies. On 3,000 consecutive test blocks the pipeline
  flags exactly the 15 anomalies, with no false alarm, in 0.4 s.
- **It has limits.** It needs complete sessions (on a truncated extract F1 drops to 0.27),
  and on BGL, a less regular system, it drops to F1 0.676 and a log-level rule does better
  (0.846), because most test windows contain templates never seen in training.
- **v1: the distillation works, the explanations do not.** The 1.5B model reproduces the
  teacher better than every baseline (ROUGE-L 0.506 on the reasoning against 0.456 for
  template retrieval) and, with batching, costs about 7 times less than calling the 70B
  teacher. But an audit against the raw logs shows that neither model finds the real cause:
  it is in other lines of the block (a write that never finished, a failed delete), not in
  the one line the model sees. A rule-based sentence built from the events missing from the
  block and the rare events it contains explains every flagged block, at no cost.
- **v2: with the whole block as input, the small model finds the cause.** Scored against
  the raw log on 123 test blocks, the fine-tuned 1.5B model names the right cause for 63% of
  the anomalous blocks, close to its 120B teacher (67%), and 100% for the three most common
  failures (write never completed: 0% in v1; failed delete; redundant addStoredBlock). Its
  remaining errors are its teacher's: it treats normal block deletions as anomalies and
  misses empty packets.

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

## Step 2, v1: explaining from one line

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

### Audit (v1): do the explanations find the real cause?

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

## Step 2, v2: explaining from the whole block

The v1 audit showed what was missing: the model only saw one line. v2 changes the input, the
annotations and the evaluation, and keeps the same small model and the same LoRA recipe.

**Input.** A summary of the whole block: every event type with its count, in order of first
appearance, the raw WARN and exception lines, and the five events a normal write always has.
The model is not told the label: it has to say itself whether there is a problem.

**Output.** JSON in English: `anomalous` (true/false), `cause`, `evidence` (the events it
relies on, `missing: ...` for an absent usual event) and a one or two sentence `explanation`.

**Data** (`src/build_dataset_v2.py`). Same chronological split as the detector: 409 training
blocks from the past, 123 test blocks from the future. Anomalous blocks are sampled per kind
of failure (up to 45 per kind in training, 15 in test), so rare failures are present. Normal
blocks are half random and half re-replication blocks, the pattern behind all of the
detector's false alarms, to check whether a model invents a problem.

**Teacher** (`src/annotate_v2.py`). GPT-OSS 120B on Groq, temperature 0, one block per call,
JSON output (Llama 3.3-70B, the v1 teacher, was not available on my account).

**Student** (`src/train_v2.py`). Qwen2.5-1.5B-Instruct, LoRA r=16 on q/k/v/o, loss on the
answer only, 5 epochs. The inputs are longer than in v1, so on a T4: batch 1 with 16
accumulation steps and gradient checkpointing. Validation perplexity goes from 1.28 to 1.16.

**Evaluation** (`src/eval_v2.py`). Every system is scored against the raw log, not against
the teacher. The gold cause of each anomalous block comes from the same rules as the pipeline
sentence. An answer is matched to a cause mainly through the events it cites as evidence
(much more reliable than its wording), plus a few keywords. I checked this matching by hand on
all 123 teacher answers: the 57 counted correct are correct, the 26 counted wrong are real
mistakes.

| System | Valid JSON | Correct cause | Wrong or vague | Missed | Invents on normal (random) | Invents on normal (re-replication) |
|---|---:|---:|---:|---:|---:|---:|
| Rules (pipeline sentence) | 100% | 100% | 0% | 0% | 60% | 100% |
| Teacher, GPT-OSS 120B | 100% | 67% | 13% | 19% | 40% | 0% |
| v1 student (one line) | 99% | 39% | 61% | 0% | - | - |
| Qwen2.5-1.5B, no fine-tuning | 27% | 4% | 0% | 96% | 0% | 0% |
| **v2 student (block summary)** | **100%** | **63%** | 24% | 13% | 40% | 0% |

Correct cause per kind of failure:

| Failure | n | Teacher | v1 | v2 |
|---|---:|---:|---:|---:|
| write never completed | 15 | 100% | 0% | **100%** |
| failed delete | 15 | 100% | 13% | **100%** |
| redundant addStoredBlock | 15 | 100% | 27% | **100%** |
| block of no file | 5 | 100% | 20% | 60% |
| replication time-out | 3 | 100% | 100% | 100% |
| empty packet | 15 | 0% | 47% | 7% |
| re-replication | 15 | 20% | 100% | 0% |

What this shows:

- **The input was the problem.** With the same model size, the most common failure (a write
  that never completed) goes from 0% in v1 to 100% in v2, and failed deletes and redundant
  addStoredBlock from 13% and 27% to 100%.
- **The small model gets close to its teacher**: 63% against 67%, with 80 times fewer
  parameters, and 98% of the evidence it cites is really in its input.
- **Fine-tuning is necessary.** The same model without it produces valid JSON 27% of the time.
- **The student inherits its teacher's blind spots.** Both treat normal block deletions as a
  problem (all 8 problems v2 invents on normal blocks are "block deletion"), and both miss
  empty packets, which are INFO lines with no error. The next step is better annotations,
  not a bigger student.
- **The 0% on re-replication normal blocks is not discrimination**: v2 never calls a
  re-replication a problem, including when it is one (0% on that category). v1 scores well on
  re-replication and empty packets only because, for those blocks, the line it gets is the
  anomalous event itself.
- **Rules and model are complementary.** The rules never miss the cause but call every
  re-replication a problem; the model is never fooled by re-replications but has blind spots.
  A combination (rules for the cause, the model for the wording and to discard benign
  re-replications) is the natural next step.

Full numbers, predictions of every system and side-by-side examples: [`results/v2/`](results/v2/).
The adapter is in `modele_hdfs_v2/`, the annotations in `data/v2/`. `v2_colab.ipynb` runs the
whole v2 (annotation, training, evaluation) on Colab.

## Putting both together

`src/pipeline.py` takes a raw log file, scores every block with the logistic regression, and
for each flagged block:

- lists the **missing events** (templates that at least 90% of normal training blocks
  contain but this block does not) and the **rare events** (present here, seen in less than
  5% of normal blocks)
- turns them into **one plain-English sentence** with fixed rules, for example "The block was
  allocated but never stored: the events a normal write ends with (addStoredBlock,
  PacketResponder terminating) never appear. The block also logged java.io.IOException:
  Could not read from stream."
- picks the line that weighs the most in the decision (a template never seen in training if
  there is one, otherwise the one with the largest weight) and sends it to the fine-tuned
  model with the label "Anomaly"

The output is a JSONL report with the block, its score, the sentence, the missing and rare
events, the chosen line and the model's explanation.

On the 1,744 blocks flagged in the test period of the full log, every block gets a specific
sentence:

| Sentence (short) | Flagged blocks |
|---|---:|
| allocated but never stored | 1,034 |
| a delete failed (BlockInfo not found) | 247 |
| replica reported as stored twice | 167 |
| empty packet while writing | 150 |
| block copied to another DataNode (re-replication) | 138 |
| addStoredBlock for a block of no file | 5 |
| pending replication timed out | 3 |

571 of them (33%) also name the Java exception the block logged, which is the closest the
log gets to a root cause. The sentences describe what the log shows, not why it happened
(a crashed node or a network problem leave the same trace), and the rules are written for
HDFS. Of the 138 re-replication blocks, 79 are false alarms of the detector: re-replication
also happens in normal blocks.

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

These numbers are for the v1 model (one line in, about 160 tokens). v2 inputs are a whole
block summary, about three times longer, so both the local model and the API cost more per
explanation; I did not re-run the benchmark for v2. The conclusion about batching holds.

## Limitations

- **v1 justifies a label, it does not find a cause.** Labels are per block and the model sees
  one line. The audit confirms it: 0 of 250 student explanations mention the real evidence.
  v2 fixes this by changing the input, but inherits its teacher's blind spots (normal
  deletions seen as anomalies, empty packets missed).
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

What I would do next: improve the v2 annotations where the teacher is wrong (tell it that
deletions are part of a normal lifecycle and that an empty packet is a problem, or correct
those labels by hand), combine the rules and the model, and replace the regex masking with a
real parser (Drain) so that detection holds on systems like BGL.

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
results/v2/            v2 evaluation: metrics, predictions of every system, examples
data/v2/               v2 blocks (summaries) and teacher annotations
modele_hdfs_v2/        v2 LoRA adapter
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
src/blocks_v2.py       v2 block summary, gold categories and scoring
src/build_dataset_v2.py  v2 train/test blocks from the full log
src/annotate_v2.py     v2 teacher annotations (Groq or Gemini API, several keys)
src/train_v2.py        v2 LoRA training
src/eval_v2.py         v2 evaluation of rules, teacher, v1, base model and v2
paper/                 write-up of the whole project (LaTeX source and PDF)
evaluation_colab.ipynb runs the evaluation on a Colab T4
pipeline_benchmark_colab.ipynb  runs the pipeline and the benchmark on a Colab T4
v2_colab.ipynb         runs the whole v2 on Colab
export_web_colab.ipynb exports the v2 model to 4-bit ONNX for the browser demo (docs/)
docs/index.html        the demo page (GitHub Pages)
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

# v2 (block-level explanations)
python src/build_dataset_v2.py --log path/to/HDFS.log --labels path/to/anomaly_label.csv
GROQ_API_KEY=... python src/annotate_v2.py --split train   # then --split test
python src/train_v2.py                   # GPU
python src/eval_v2.py                    # GPU for v1, base and v2

# live demo (GPU, e.g. Colab; --share prints a public link)
pip install gradio "huggingface-hub<1.0"
python src/app.py --share
```
