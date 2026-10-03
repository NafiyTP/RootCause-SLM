"""
Full pipeline on a raw HDFS log file: detect anomalous blocks, then explain them.

1. group the lines by block id and score each block with the logistic regression
   trained by src/detect.py
2. for every flagged block, pick the line that best explains the decision: a template
   never seen in training first, otherwise the template with the largest weight
3. list the events that almost every normal block has (90% or more of the normal
   training blocks) but this block does not. Many HDFS anomalies are writes that never
   finish, so what is missing says more than any single line
4. send the chosen line to the fine-tuned Qwen2.5-1.5B with the label "Anomaly"

The model still explains one line, not the whole block (that is how it was trained),
but the label now comes from a detector instead of the ground truth.

Usage:
    python src/detect.py --log HDFS.log --labels anomaly_label.csv   # train the detector first
    python src/pipeline.py --log some_logs.log --top 10              # detect + explain
    python src/pipeline.py --log some_logs.log --no-llm              # detection only, CPU
"""

import argparse
import json
import os
import pickle
import time
from collections import defaultdict

import numpy as np

from detect import BLK_RE, MODEL_PATH, ROOT, CountVectorizer
from evaluate import template

OUT_PATH = os.path.join(ROOT, "results", "pipeline", "report.jsonl")


def load_detector(path=MODEL_PATH):
    """Returns the vocabulary, the weights (one per template + <UNK>) and the intercept."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    vec = CountVectorizer().fit([d["templates"]])  # rebuild the same vocabulary
    return vec, np.array(d["coef"]), d["intercept"], d.get("normal_freq", {})


def predict_proba(X, coefs, intercept):
    """Same as LogisticRegression.predict_proba(np.log1p(X))[:, 1]."""
    return 1.0 / (1.0 + np.exp(-(np.log1p(X) @ coefs + intercept)))


def read_blocks(log_path):
    """{block_id: [(template, raw line), ...]}"""
    blocks = defaultdict(list)
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            for b in set(BLK_RE.findall(line)):
                blocks[b].append((template(line), line))
    return blocks


def missing_events(lines, normal_freq, min_freq=0.9):
    """Templates found in at least min_freq of the normal blocks but absent from this one."""
    present = {t for t, _ in lines}
    return [t for t, f in sorted(normal_freq.items(), key=lambda kv: -kv[1])
            if f >= min_freq and t not in present]


def pick_line(lines, vec, coefs):
    """The line whose template pushes the most towards 'anomaly'. Unseen templates win."""
    def weight(t):
        i = vec.index.get(t)
        return np.inf if i is None else coefs[i]
    return max(lines, key=lambda tl: weight(tl[0]))[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=10, help="explain the k highest-scoring blocks")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    vec, coefs, intercept, normal_freq = load_detector()

    t0 = time.perf_counter()
    blocks = read_blocks(args.log)
    ids = list(blocks)
    X = vec.transform([[t for t, _ in blocks[b]] for b in ids])
    scores = predict_proba(X, coefs, intercept)
    print(f"{len(ids):,} blocks scored in {time.perf_counter() - t0:.1f}s")

    flagged = sorted([(s, b) for s, b in zip(scores, ids) if s > args.threshold], reverse=True)
    print(f"{len(flagged)} blocks flagged, explaining the top {min(args.top, len(flagged))}")

    model = tokenizer = None
    if not args.no_llm and flagged:
        from inference import explain, load_model  # imports torch, only when needed
        model, tokenizer = load_model()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rank, (score, b) in enumerate(flagged):
            line = pick_line(blocks[b], vec, coefs)
            row = {"block_id": b, "score": round(float(score), 4),
                   "n_lines": len(blocks[b]), "line": line,
                   "missing": missing_events(blocks[b], normal_freq)}
            if model is not None and rank < args.top:
                t1 = time.perf_counter()
                r = explain(line, "Anomaly", model, tokenizer)
                row.update(cause=r["cause"], raisonnement=r["raisonnement"],
                           explain_s=round(time.perf_counter() - t1, 2))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if rank < args.top:
                print(f"\n[{score:.2f}] {b} ({len(blocks[b])} lines)")
                print(f"  line:  {line[:140]}")
                for t in row["missing"]:
                    print(f"  missing: {t}")
                if "cause" in row:
                    print(f"  cause: {row['cause']}")
                    print(f"  reasoning: {row['raisonnement']}")
    print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
