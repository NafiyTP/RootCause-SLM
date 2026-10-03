"""
Full pipeline on a raw HDFS log file: detect anomalous blocks, then explain them.

1. group the lines by block id and score each block with the logistic regression
   trained by src/detect.py
2. for every flagged block, list the missing events (templates that 90% or more of the
   normal training blocks contain but this block does not) and the rare events (present
   here, seen in less than 5% of normal blocks). Many HDFS anomalies are writes that
   never finish, so what is missing says more than any single line
3. turn them into one plain-English sentence with fixed rules (RULES below). No model is
   involved, so the sentence is always well formed and matches the log
4. optionally, pick the line that weighs most in the score (a template never seen in
   training first) and send it to the fine-tuned Qwen2.5-1.5B with the label "Anomaly".
   The model only sees that line, which is why its explanation is not reliable (see
   src/audit_explanations.py)

Usage:
    python src/detect.py --log HDFS.log --labels anomaly_label.csv   # train the detector first
    python src/pipeline.py --log some_logs.log --top 10              # detect + explain
    python src/pipeline.py --log some_logs.log --no-llm              # detection only, CPU
"""

import argparse
import json
import os
import pickle
import re
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


def read_lines(lines):
    """{block_id: [(template, raw line), ...]} from an iterable of raw lines."""
    blocks = defaultdict(list)
    for line in lines:
        line = line.strip()
        for b in set(BLK_RE.findall(line)):
            blocks[b].append((template(line), line))
    return blocks


def read_blocks(log_path):
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        return read_lines(f)


def missing_events(lines, normal_freq, min_freq=0.9):
    """Templates found in at least min_freq of the normal blocks but absent from this one."""
    present = {t for t, _ in lines}
    return [t for t, f in sorted(normal_freq.items(), key=lambda kv: -kv[1])
            if f >= min_freq and t not in present]


def rare_events(lines, normal_freq, max_freq=0.05):
    """Templates present in this block but found in less than max_freq of the normal blocks."""
    seen, out = set(), []
    for t, _ in lines:
        if t not in seen and normal_freq.get(t, 0) < max_freq:
            seen.add(t)
            out.append(t)
    return out


# (what to look for, sentence). Checked in this order; the first match wins.
RULES = [
    (lambda missing, rare: any("addStoredBlock: blockMap updated" in t for t in missing),
     "The block was allocated but never stored: the events a normal write ends with "
     "(addStoredBlock, PacketResponder terminating) never appear."),
    (lambda missing, rare: any("Unexpected error trying to delete" in t for t in rare),
     "A delete failed: the DataNode had no record of the block (BlockInfo not found)."),
    (lambda missing, rare: any("does not belong to any file" in t for t in rare),
     "The NameNode received addStoredBlock for a block that belongs to no file."),
    (lambda missing, rare: any("Redundant addStoredBlock" in t for t in rare),
     "The same replica was reported as stored twice (redundant addStoredBlock)."),
    (lambda missing, rare: any("empty packet" in t for t in rare),
     "A DataNode received an empty packet while writing the block."),
    (lambda missing, rare: any("timed out block" in t for t in rare),
     "A pending replication of the block timed out."),
    (lambda missing, rare: any("to replicate" in t or "Starting thread to transfer" in t for t in rare),
     "The NameNode had to copy the block to another DataNode (re-replication), which "
     "usually means a replica was lost or missing."),
    (lambda missing, rare: bool(rare),
     "The block contains events that normal blocks almost never have."),
    (lambda missing, rare: bool(missing),
     "Some events that normal blocks have are absent."),
]


def exceptions(rare):
    """Exception names that appear in the rare events, e.g. 'java.io.IOException: Could not read from stream'."""
    out = []
    for t in rare:
        m = re.search(r"((?:java|javax)\.[\w.]*(?:Exception|Error)\b(?::[^<]*)?)", t)
        if m and m.group(1).strip() not in out:
            out.append(m.group(1).strip())
    return out


def describe(missing, rare):
    """One plain-English sentence from the missing and rare events. No model involved."""
    sentence = "No single event stands out; the counts as a whole look unusual."
    for test, s in RULES:
        if test(missing, rare):
            sentence = s
            break
    exc = exceptions(rare)
    if exc:
        sentence += " The block also logged " + "; ".join(exc) + "."
    return sentence


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
            missing = missing_events(blocks[b], normal_freq)
            rare = rare_events(blocks[b], normal_freq)
            row = {"block_id": b, "score": round(float(score), 4),
                   "n_lines": len(blocks[b]), "summary": describe(missing, rare),
                   "missing": missing, "rare": rare, "line": line}
            if model is not None and rank < args.top:
                t1 = time.perf_counter()
                r = explain(line, "Anomaly", model, tokenizer)
                row.update(cause=r["cause"], raisonnement=r["raisonnement"],
                           explain_s=round(time.perf_counter() - t1, 2))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if rank < args.top:
                print(f"\n[{score:.2f}] {b} ({len(blocks[b])} lines)")
                print(f"  {row['summary']}")
                for t in missing:
                    print(f"  missing: {t}")
                for t in rare:
                    print(f"  rare:    {t}")
                print(f"  line:  {line[:140]}")
                if "cause" in row:
                    print(f"  cause: {row['cause']}")
                    print(f"  reasoning: {row['raisonnement']}")
    print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
