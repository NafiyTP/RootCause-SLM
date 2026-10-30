# RootCause-SLM

I wanted to know if a small model running on my own GPU could explain anomalies in server logs.
In a company you often can't send infrastructure logs to an external API, so a local model
would be useful. The project ended up with two parts:

1. Detection: find the anomalous blocks in HDFS logs. This is the main part, and it works well.
2. Explanation: fine-tune Qwen2.5-1.5B with LoRA to say why a block is anomalous, with a big
   model as the teacher. This part is more of an experiment.

Demo: https://nafiytp.github.io/RootCause-SLM/

It runs in the browser. You can paste HDFS lines, add or remove events in a block and see the
score change, move the detection threshold, and read the saved answers of the models on the
123 v2 test blocks.

## Short version

- A logistic regression on template counts gets F1 0.973 on the full HDFS_v1 log (11M lines,
  575k blocks, chronological split). Looking only at the log level finds a quarter of the
  anomalies. On 3,000 consecutive test blocks the pipeline flags the 15 anomalies and nothing
  else, in 0.4 s.
- It needs complete sessions: on a truncated extract F1 falls to 0.27. On BGL, a less regular
  system, it gets F1 0.676 and a simple log-level rule does better (0.846).
- In v1 the explanation model saw only one line of the block. It copies its 70B teacher better
  than every baseline, but when I checked against the raw logs, neither of them finds the real
  cause. The cause is in other lines of the block.
- In v2 the model sees a summary of the whole block. Then the fine-tuned 1.5B model gives the
  right cause for 63% of the anomalous blocks, and its 120B teacher for 67%.

## Data

Logs and labels come from [Loghub](https://github.com/logpai/loghub) (HDFS_v1: Hadoop jobs on
more than 200 Amazon EC2 nodes in 2008, about 11M lines). The labels are per block, in
`anomaly_label.csv`. The raw log is too big for the repo, so `src/detect.py` reads it from
where you downloaded it.

For the v1 explanation model, each line gets the label of its block. Llama 3.3-70B only writes
the `cause` and `raisonnement` fields. It never decides the label. The annotations are in
French, so the prompts in the code are in French too.

| Split | Lines | Normal / Anomaly |
|---|---:|---|
| Train (`data/hdfs_dataset.json`), 85/15 train/val | 1,999 | 1,930 / 69 |
| Test (`data/hdfs_test_dataset.json`) | 527 | 277 / 250 |

The test lines come from blocks that are never in the training set. I oversampled the
anomalies so the metrics actually say something about them.

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

A block is a session: all the lines that contain its id. I turn each line into a template by
masking block ids, IPs, paths and numbers, and each block becomes a vector of template counts.
Templates never seen in training go into one `<UNK>` column. I don't drop them, because a new
kind of message is exactly what you want to notice.

The split is chronological. Blocks are sorted by their first line, the first 80% are for
training and the last 20% for testing. With a random split the model could train on blocks
that come after the test ones.

I compare three detectors:

- WARN/ERROR rule: a block is anomalous if one of its lines is not INFO. This is what you would
  do without any ML.
- PCA (unsupervised, the classic method on this dataset, Xu et al. 2009): fit PCA on normalized
  counts, keep 95% of the variance, flag blocks with a large reconstruction error. The threshold
  is the 97th percentile of the training scores.
- Logistic regression on log(1 + counts), with `class_weight="balanced"` since anomalies are
  about 3% of the blocks.

Results on the full HDFS_v1 log (11,175,629 lines, 575,061 blocks, 54 templates; the test set
is the last 115,013 blocks, 1.5% anomalies):

| Detector | Precision | Recall | F1 | Flagged |
|---|---:|---:|---:|---:|
| WARN/ERROR rule | 1.000 | 0.248 | 0.398 | 417 |
| PCA (unsupervised) | 0.379 | 0.051 | 0.090 | 227 |
| Logistic regression | 0.955 | 0.992 | 0.973 | 1,744 |

The rule never gives a false alarm, but three quarters of the anomalous blocks only have INFO
lines.

PCA ranks the blocks well (ROC AUC 0.98 on the test set) but its threshold does not carry over.
It is set on the training period, where 3.3% of the blocks are anomalies, and in the test
period there are only 1.5%. To set it correctly you need labels or a known anomaly rate.

The logistic regression finds almost every anomaly. The templates with the biggest weights make
sense: a redundant `addStoredBlock`, a block that belongs to no file, a failed delete, a
replication timeout.

One thing I learned the hard way: the detector needs complete sessions. On an extract of the
first 105k lines the same model only gets F1 0.27. The most recent blocks are cut at the end of
the extract, so they look like writes that never finished, and they get flagged. On a live
stream you would have to wait until a block is closed before scoring it.

Detection runs at about 115k lines/s on one CPU core, parsing included (about 2 minutes for the
full log). The detector is saved in `modele_detection/detector.pkl` as plain lists (templates,
weights, and how often each template shows up in normal blocks), so it loads with any
scikit-learn version. Full numbers in [`results/detection/`](results/detection/).

### Does it work on another system? BGL

HDFS is an easy dataset: 54 templates and very regular sessions. So I ran the same detectors on
BGL (Blue Gene/L supercomputer, also from Loghub, 4.7M lines) with `src/detect_bgl.py`. BGL has
no session id, so a session is a 60-minute window over the whole machine, anomalous if it has at
least one alert line. The level rule flags a window if one of its lines is FATAL, FAILURE,
SEVERE or ERROR. Chronological split, the test set is the last 724 windows (18.6% anomalous):

| Detector | Precision | Recall | F1 | Flagged |
|---|---:|---:|---:|---:|
| Level rule (FATAL/FAILURE/SEVERE/ERROR) | 0.734 | 1.000 | 0.846 | 184 |
| PCA (unsupervised) | 0.203 | 0.541 | 0.295 | 359 |
| Logistic regression | 0.972 | 0.518 | 0.676 | 72 |

Here it's the opposite of HDFS: the simple rule wins. The logistic regression is still precise
but misses half of the anomalous windows. The problem is my masking. I wrote it for HDFS, and on
BGL it leaves 13,603 templates in training. 412 of the 724 test windows (57%) contain a template
never seen before, and they all end up in the same `<UNK>` column, so a new kind of failure looks
almost like a new kind of normal message. To fix this I would need a real log parser (Drain for
example) and a model that handles new templates. I didn't do that here. Full numbers in
[`results/detection_bgl/`](results/detection_bgl/).

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

I compare the fine-tuned model with three simple baselines and with the base model in zero-shot
and few-shot, on the 527 test lines. Decoding is greedy, and the fine-tuned model gets exactly
the training prompt. The references are the Llama 3.3-70B annotations, so these scores say how
close each system is to the teacher, not if the cause is actually right.

| System | Valid JSON | Cause exact match | ROUGE-L cause | ROUGE-L reasoning | BERTScore reasoning |
|---|---:|---:|---:|---:|---:|
| Most frequent cause | 100% | 0.461 | 0.594 | 0.347 | 0.797 |
| Majority cause per label | 100% | 0.461 | 0.589 | 0.356 | 0.783 |
| Template retrieval | 100% | 0.548 | 0.860 | 0.456 | 0.825 |
| Qwen2.5-1.5B zero-shot | 6.5% | 0.000 | 0.008 | 0.017 | 0.048 |
| Qwen2.5-1.5B few-shot (3 examples) | 100% | 0.000 | 0.393 | 0.292 | 0.778 |
| Qwen2.5-1.5B + LoRA | 100% | 0.617 | 0.874 | 0.506 | 0.836 |

The baseline to beat is template retrieval: for each test line it copies the annotation of a
training line with the same template and label. The fine-tuned model is +0.050 ROUGE-L above it
on the reasoning (paired bootstrap 95% CI [0.038, 0.064]) and doubles the exact cause match on
anomalies (0.284 vs 0.140). On the reasoning of anomalous lines they are tied though (0.455 vs
0.461).

In zero-shot the base model mostly fails on the format: it puts the JSON in code fences, breaks
strings over several lines or runs out of tokens. With three examples in the prompt the format is
fine, but it paraphrases the causes and never matches the reference wording.

Per-label numbers, confidence intervals, all predictions and examples are in
[`results/eval_n527/`](results/eval_n527/).

(An older version of this README had numbers from a 20-example check with only Normal lines,
because of a resume bug in the old evaluation script. The table above replaces them.)

### Audit: do the v1 explanations find the real cause?

The scores above only compare to the teacher. To check against the logs themselves,
`src/audit_explanations.py` rebuilds the whole block of each anomalous test line from `HDFS.log`
and looks for the evidence at the block level: events that 90% of normal blocks have and this one
doesn't, or rare events it contains. Then it checks if the cause or the reasoning mentions that
evidence. I also read 50 of them by hand (25 anomalies, 25 normal) to make sure the keyword check
wasn't missing anything.

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

Both models basically tell one of two stories depending on the line they get. An `allocateBlock`
line becomes "Erreur d'allocation de bloc", a `Receiving block` line becomes "Connexion non
autorisée". The logs support neither: the allocation worked, and nothing in the block points to
an unauthorized connection. On normal lines the explanations are fine, since they just describe
the event. So the fine-tuning did its job (the student copies the teacher well), but what it
copies is a guess, because the cause is not in the input. Details and the hand-checked sample
are in [`results/audit/`](results/audit/).

## Step 2, v2: explaining from the whole block

After the audit the problem was clear: the model only saw one line. In v2 I changed the input,
the annotations and the evaluation, and kept the same model and the same LoRA setup.

Input: a summary of the whole block. Every event type with its count, in order of first
appearance, the raw WARN and exception lines, and the five events a normal write always has. The
model is not given the label, it has to decide itself if there is a problem.

Output: JSON in English with `anomalous` (true/false), `cause`, `evidence` (the events it relies
on, `missing: ...` for a usual event that is absent) and a short `explanation`.

Data (`src/build_dataset_v2.py`): same chronological split as the detector, 409 training blocks
from the past and 123 test blocks from the future. Anomalous blocks are sampled per type of
failure (up to 45 per type in training, 15 in test) so the rare ones are there too. Normal blocks
are half random and half re-replication blocks. Those are behind all the detector's false alarms,
so I wanted to see if a model invents a problem on them.

Teacher (`src/annotate_v2.py`): GPT-OSS 120B on Groq, temperature 0, one block per call, JSON
output. Llama 3.3-70B, the v1 teacher, wasn't available on my account anymore.

Student (`src/train_v2.py`): Qwen2.5-1.5B-Instruct, LoRA r=16 on q/k/v/o, loss on the answer
only, 5 epochs. The inputs are longer than in v1, so to fit on a T4 I used batch 1 with 16
accumulation steps and gradient checkpointing. Validation perplexity goes from 1.28 to 1.16.

Evaluation (`src/eval_v2.py`): every system is scored against the raw log, not against the
teacher. The gold cause of each anomalous block comes from the same rules as the pipeline
sentence (see below). An answer is matched to a cause mostly through the events it cites as
evidence, which is much more reliable than its wording, plus a few keywords. I checked this
matching by hand on all 123 teacher answers: the 57 counted correct are correct, and the 26
counted wrong are real mistakes.

| System | Valid JSON | Correct cause | Wrong or vague | Missed | Invents on normal (random) | Invents on normal (re-replication) |
|---|---:|---:|---:|---:|---:|---:|
| Rules (pipeline sentence) | 100% | 100% | 0% | 0% | 60% | 100% |
| Teacher, GPT-OSS 120B | 100% | 67% | 13% | 19% | 40% | 0% |
| v1 student (one line) | 99% | 39% | 61% | 0% | - | - |
| Qwen2.5-1.5B, no fine-tuning | 27% | 4% | 0% | 96% | 0% | 0% |
| v2 student (block summary) | 100% | 63% | 24% | 13% | 40% | 0% |

Correct cause per type of failure:

| Failure | n | Teacher | v1 | v2 |
|---|---:|---:|---:|---:|
| write never completed | 15 | 100% | 0% | 100% |
| failed delete | 15 | 100% | 13% | 100% |
| redundant addStoredBlock | 15 | 100% | 27% | 100% |
| block of no file | 5 | 100% | 20% | 60% |
| replication time-out | 3 | 100% | 100% | 100% |
| empty packet | 15 | 0% | 47% | 7% |
| re-replication | 15 | 20% | 100% | 0% |

So the input was the real issue. With the same model size, the most common failure (a write that
never completed) goes from 0% in v1 to 100% in v2, and failed deletes and redundant
addStoredBlock go from 13% and 27% to 100%. The 1.5B model gets 63% against 67% for a teacher
with 80 times more parameters, and 98% of the evidence it cites is really in its input. Without
fine-tuning, the same model only gives valid JSON 27% of the time.

The student also copies its teacher's mistakes. Both see normal block deletions as a problem (all
8 problems v2 invents on normal blocks are "block deletion"), and both miss empty packets, which
are INFO lines with no error. So the next thing to fix is the annotations, not the size of the
student.

Be careful with the 0% on re-replication normal blocks: it doesn't mean v2 tells them apart, it
just never calls a re-replication a problem, even when it is one (0% on that category). And v1
only does well on re-replication and empty packets because for those blocks the one line it gets
is the anomalous event itself.

The rules and the model fail in different places. The rules always find the cause but call every
re-replication a problem. The model is never fooled by re-replications but has blind spots. So
combining them (rules for the cause, the model for the wording and to drop harmless
re-replications) would probably be better than either one.

Full numbers, all predictions and examples: [`results/v2/`](results/v2/). The adapter is in
`modele_hdfs_v2/` and the annotations in `data/v2/`. `v2_colab.ipynb` runs all of v2
(annotation, training, evaluation) on Colab.

## Detection + explanation together

`src/pipeline.py` takes a raw log file, scores every block with the logistic regression, and for
each flagged block:

- lists the missing events (templates that at least 90% of normal training blocks have but this
  block doesn't) and the rare events (present here, seen in less than 5% of normal blocks)
- turns them into one plain-English sentence with fixed rules, for example "The block was
  allocated but never stored: the events a normal write ends with (addStoredBlock,
  PacketResponder terminating) never appear. The block also logged java.io.IOException: Could
  not read from stream."
- picks the line that weighs the most in the decision (a template never seen in training if there
  is one, otherwise the one with the biggest weight) and sends it to the fine-tuned model with
  the label "Anomaly"

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

571 of them (33%) also name the Java exception the block logged, which is as close to a root
cause as the log gets. The sentences describe what the log shows, not why it happened (a crashed
node and a network problem leave the same trace), and the rules only work for HDFS. Of the 138
re-replication blocks, 79 are false alarms from the detector, since re-replication also happens
in normal blocks.

I also tested it on 3,000 consecutive blocks from the test period, with all their lines (38,957
lines, 15 anomalous blocks). It flagged exactly the 15 anomalies, no false alarm, in 0.4 s. For
12 of them the missing events explain what happened: the block was allocated and the transfer
started, but it was never stored (no `addStoredBlock`, no `PacketResponder terminating`). For
those 12 the line sent to the model is the `allocateBlock` line, and from that line alone it can
only guess. For the other 3, the chosen line is the anomalous event itself (a redundant
`addStoredBlock`, an empty packet), so the line selection works when the anomaly is something
that happened, not something that didn't happen. For the first kind, the missing-events list is
the better explanation, and it's free.

## Speed and cost

`src/benchmark.py` measures the fine-tuned model on a T4 (fp16, LoRA merged into the weights),
one request at a time and with batching. Then it compares the cost per 1,000 explanations with
calling Llama 3.3-70B on Groq ($0.59 / $0.79 per million input / output tokens). The teacher cost
is estimated from the token counts of the same prompts and answers.

Measured on 64 test lines (161 tokens in and 92 out on average), with a T4 at $0.35 per hour:

| Setup | s / line | lines / s | tokens / s | USD per 1k lines |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B + LoRA, batch 1 | 2.918 | 0.34 | 30 | 0.284 |
| Qwen2.5-1.5B + LoRA, batch 16 | 0.251 | 3.99 | 344 | 0.024 |
| Llama 3.3-70B on Groq (estimate) | | | | 0.168 |

One request at a time, the small model is slow (about 3 s per explanation) and actually more
expensive than the API, because you pay for the GPU while it waits. With batches of 16 it's about
12 times faster and about 7 times cheaper than the teacher, and it uses 3.4 GB of GPU memory. So
running it locally is only worth it when you can group explanations, which is the case when the
detector flags a lot of blocks at once. The price is for the GPU only, not the whole VM, so the
real gap is a bit smaller. Full numbers in [`results/benchmark/`](results/benchmark/).

These numbers are for the v1 model (one line in, about 160 tokens). v2 inputs are about three
times longer, so both the local model and the API cost more per explanation. I didn't re-run the
benchmark for v2, but batching should still be what makes the difference.

## Limitations

- v1 justifies a label, it doesn't find a cause: labels are per block and the model sees one
  line, and the audit confirms it (0 of 250 explanations mention the real evidence). v2 fixes
  this by changing the input but keeps its teacher's blind spots (normal deletions seen as
  anomalies, empty packets missed).
- The v1 training data is small and repetitive: 69 training anomalies over 29 different causes,
  and only 15 message templates. Validation perplexity is already 1.19 after one epoch, and on
  the test set the model only uses 8 distinct causes (the references have 19).
- There are no human-validated root causes for this dataset. The audit checks against the logs,
  but with keywords and a 50-example manual check, not with expert labels.
- The logistic regression is supervised and fails on cut blocks. PCA works without labels but its
  threshold doesn't carry over from one period to the next.
- The detector doesn't generalize as it is: on BGL the masking breaks down (13,603 templates,
  57% of test windows with an unseen one) and a log-level rule wins.

What I would do next: fix the v2 annotations where the teacher is wrong (tell it that deletions
are part of a normal lifecycle and that an empty packet is a problem, or correct those labels by
hand), combine the rules and the model, and replace my regex masking with a real parser (Drain)
so detection also works on systems like BGL.

## Repo structure

```
data/                  train and test sets for the explanation model (JSON and CSV)
data/v2/               v2 blocks (summaries) and teacher annotations
modele_hdfs/           v1 LoRA adapter, tokenizer, training history
modele_hdfs_v2/        v2 LoRA adapter
modele_detection/      trained detector (template list + logistic regression)
results/detection/     detection metrics (full HDFS_v1 log)
results/detection_bgl/ detection metrics on BGL
results/audit/         v1 explanations checked against the raw blocks
results/eval_n527/     v1 explanation metrics, predictions and examples
results/benchmark/     speed and cost of the fine-tuned model
results/v2/            v2 evaluation: metrics, predictions of every system, examples
src/detect.py          block-level detection (rule, PCA, logistic regression)
src/detect_bgl.py      the same detectors on BGL, with time windows as sessions
src/audit_explanations.py  checks the explanations against the whole block
src/dataset.py         Dataset and collator (ChatML, loss masking, dynamic padding)
src/train.py           v1 LoRA training with the weighted sampler
src/evaluate.py        v1 baselines, zero-shot, few-shot and fine-tuned evaluation
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
docs/index.html        the demo page (GitHub Pages)
evaluation_colab.ipynb runs the v1 evaluation on a Colab T4
pipeline_benchmark_colab.ipynb  runs the pipeline and the benchmark on a Colab T4
v2_colab.ipynb         runs all of v2 on Colab
export_web_colab.ipynb exports the v2 model to 4-bit ONNX for the browser
```

## Usage

```bash
pip install -r requirements.txt

# detection (CPU), needs HDFS.log and anomaly_label.csv from Loghub HDFS_v1
python src/detect.py --log path/to/HDFS.log --labels path/to/anomaly_label.csv
python src/detect_bgl.py --log path/to/BGL.log
python src/audit_explanations.py --log path/to/HDFS.log

# v1 explanation model
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

# Gradio demo with the live model (GPU, e.g. Colab; --share prints a public link)
pip install gradio "huggingface-hub<1.0"
python src/app.py --share
```
