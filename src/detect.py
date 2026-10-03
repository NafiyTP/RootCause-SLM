"""
Block-level anomaly detection on raw HDFS logs.

The fine-tuned model only explains a label, it never decides it. This script is the
missing first stage: it groups the raw log lines by block id (one block = one session),
turns each block into a vector of template counts, and classifies the block.

Detectors compared:
  - WARN/ERROR rule: a block is anomalous if one of its lines is not INFO
  - PCA (unsupervised, as in Xu et al. 2009): large reconstruction error = anomaly
  - logistic regression (supervised) on log(1 + counts)

The split is chronological: blocks are sorted by their first line, the first 80% are
used for training and the last 20% for testing, so the model never sees the future.
Templates that only show up in the test part fall into one <UNK> column.

Input: HDFS.log and anomaly_label.csv from Loghub HDFS_v1 (not in the repo, too big).

Usage:
    python src/detect.py --log path/to/HDFS.log --labels path/to/anomaly_label.csv
    python src/detect.py --log ... --labels ... --max-lines 1000000   # faster test
"""

import argparse
import csv
import json
import os
import pickle
import re
import time
from collections import Counter, defaultdict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from evaluate import template  # same template function as the baselines

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.join(SRC_DIR, "..")
OUT_DIR  = os.path.join(ROOT, "results", "detection")
MODEL_PATH = os.path.join(ROOT, "modele_detection", "detector.pkl")

BLK_RE = re.compile(r"blk_-?\d+")


# ---------------------------------------------------------------- sessions

def read_sessions(log_path, max_lines=None):
    """
    Returns {block_id: [template, ...]} in order of first appearance, plus the level
    (INFO/WARN/...) seen in each block. A line that mentions several blocks counts for each.
    """
    sessions = defaultdict(list)
    levels = defaultdict(set)
    n = 0
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if max_lines and n >= max_lines:
                break
            n += 1
            blocks = set(BLK_RE.findall(line))
            if not blocks:
                continue
            t = template(line.strip())
            parts = line.split(" ", 4)
            level = parts[3] if len(parts) > 3 else "INFO"
            for b in blocks:
                sessions[b].append(t)
                levels[b].add(level)
    print(f"read {n:,} lines, {len(sessions):,} blocks")
    return sessions, levels, n


def read_labels(path):
    with open(path, encoding="utf-8") as f:
        return {r["BlockId"]: int(r["Label"] == "Anomaly") for r in csv.DictReader(f)}


# ---------------------------------------------------------------- features

class CountVectorizer:
    """Template counts per block. Vocabulary = templates seen in training + <UNK>."""

    def fit(self, sessions):
        vocab = sorted({t for s in sessions for t in s})
        self.index = {t: i for i, t in enumerate(vocab)}
        self.unk = len(vocab)
        self.names = vocab + ["<UNK>"]
        return self

    def transform(self, sessions):
        X = np.zeros((len(sessions), len(self.names)), dtype=np.float32)
        for i, s in enumerate(sessions):
            for t, c in Counter(s).items():
                X[i, self.index.get(t, self.unk)] += c
        return X


# ---------------------------------------------------------------- detectors

class PCADetector:
    """
    Fit PCA on standardized log counts, keep the components that explain 95% of the
    variance, and score a block by the squared error of its reconstruction. The
    threshold is a percentile of the training scores; no label is used.
    """

    def __init__(self, var=0.95, percentile=97.0):
        self.var = var
        self.percentile = percentile

    def fit(self, X):
        self.scaler = StandardScaler().fit(np.log1p(X))
        Z = self.scaler.transform(np.log1p(X))
        self.pca = PCA(n_components=self.var, svd_solver="full").fit(Z)
        self.threshold = np.percentile(self.score(X), self.percentile)
        return self

    def score(self, X):
        Z = self.scaler.transform(np.log1p(X))
        rec = self.pca.inverse_transform(self.pca.transform(Z))
        return ((Z - rec) ** 2).sum(axis=1)

    def predict(self, X):
        return (self.score(X) > self.threshold).astype(int)


class LogRegDetector:
    """Logistic regression on log(1 + counts). class_weight='balanced' for the 3% anomalies."""

    def fit(self, X, y):
        self.clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
        self.clf.fit(np.log1p(X), y)
        return self

    def score(self, X):
        return self.clf.predict_proba(np.log1p(X))[:, 1]

    def predict(self, X):
        return (self.score(X) > 0.5).astype(int)


# ---------------------------------------------------------------- main

def prf(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4),
            "flagged": int(pred.sum()), "true_anomalies": int(y.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="HDFS.log from Loghub HDFS_v1")
    ap.add_argument("--labels", required=True, help="anomaly_label.csv")
    ap.add_argument("--max-lines", type=int, default=None)
    ap.add_argument("--train-frac", type=float, default=0.8)
    args = ap.parse_args()

    t0 = time.perf_counter()
    sessions, levels, n_lines = read_sessions(args.log, args.max_lines)
    t_parse = time.perf_counter() - t0
    labels = read_labels(args.labels)

    # dicts keep insertion order, so this is the order of first appearance in the log
    blocks = [b for b in sessions if b in labels]
    y = np.array([labels[b] for b in blocks])
    cut = int(len(blocks) * args.train_frac)
    tr, te = blocks[:cut], blocks[cut:]
    y_tr, y_te = y[:cut], y[cut:]
    print(f"train: {len(tr):,} blocks ({y_tr.mean():.1%} anomalies), "
          f"test: {len(te):,} blocks ({y_te.mean():.1%} anomalies)")

    vec = CountVectorizer().fit([sessions[b] for b in tr])
    X_tr = vec.transform([sessions[b] for b in tr])
    X_te = vec.transform([sessions[b] for b in te])
    unk_blocks = int((X_te[:, vec.unk] > 0).sum())
    print(f"{len(vec.names) - 1} templates in training, "
          f"{unk_blocks} test blocks contain an unseen template")

    results = {}

    # 1. rule: anything that is not INFO
    pred = np.array([int(bool(levels[b] - {"INFO"})) for b in te])
    results["WARN/ERROR rule"] = prf(y_te, pred)

    # 2. PCA, no labels
    t0 = time.perf_counter()
    pca = PCADetector().fit(X_tr)
    results["PCA (unsupervised)"] = prf(y_te, pca.predict(X_te))
    results["PCA (unsupervised)"]["fit_s"] = round(time.perf_counter() - t0, 2)

    # 3. logistic regression
    t0 = time.perf_counter()
    lr = LogRegDetector().fit(X_tr, y_tr)
    results["Logistic regression"] = prf(y_te, lr.predict(X_te))
    results["Logistic regression"]["fit_s"] = round(time.perf_counter() - t0, 2)

    # throughput of the full detection step (parsing + features + prediction)
    t0 = time.perf_counter()
    lr.predict(vec.transform([sessions[b] for b in blocks]))
    lines_per_s = n_lines / (t_parse + time.perf_counter() - t0)

    # which templates push a block towards "anomaly"
    coefs = lr.clf.coef_[0]
    top = [(vec.names[i], round(float(coefs[i]), 2)) for i in np.argsort(-coefs)[:8]]

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    # only plain Python lists (templates, weights), so the file loads with any
    # scikit-learn version, or without scikit-learn at all
    # share of normal training blocks that contain each template, so the pipeline can
    # say which usual events are missing from a flagged block
    normal = X_tr[y_tr == 0] > 0
    normal_freq = dict(zip(vec.names[:-1], normal[:, :-1].mean(axis=0).round(4).tolist()))
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"templates": vec.names[:-1],
                     "coef": lr.clf.coef_[0].tolist(),
                     "intercept": float(lr.clf.intercept_[0]),
                     "normal_freq": normal_freq}, f)

    summary = {
        "lines": n_lines, "blocks": len(blocks), "train_blocks": len(tr), "test_blocks": len(te),
        "test_anomaly_rate": round(float(y_te.mean()), 4),
        "templates": len(vec.names) - 1, "test_blocks_with_unseen_template": unk_blocks,
        "lines_per_second_cpu": int(lines_per_s),
        "top_anomaly_templates": top, "detectors": results,
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md = [f"# Block-level detection ({n_lines:,} lines, {len(blocks):,} blocks)", "",
          f"Chronological split, test = last {len(te):,} blocks "
          f"({y_te.mean():.1%} anomalies).", "",
          "| Detector | Precision | Recall | F1 | Flagged |",
          "|---|---:|---:|---:|---:|"]
    for name, r in results.items():
        md.append(f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} "
                  f"| {r['flagged']} |")
    md += ["", f"Detection throughput on CPU: about {lines_per_s:,.0f} lines/s "
           "(parsing included).", "", "## Templates with the largest positive weight", ""]
    md += [f"- {w:+.2f}  `{t}`" for t, w in top]
    with open(os.path.join(OUT_DIR, "metrics.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"\ndetector saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
