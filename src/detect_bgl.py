"""
Does the detection step carry over to another system? Same detectors as src/detect.py,
on BGL (Blue Gene/L supercomputer logs, Loghub).

BGL has no block id, so a session is a fixed time window (60 minutes by default) over
the whole machine. Each line carries its own label in the first field ("-" = normal,
anything else = alert category); a window is anomalous if it contains at least one
alert line. Templates: component + level + message with numbers, hex values, IPs and
paths masked. The split is chronological (first 80% of the windows for training).

The level rule plays the role of the WARN/ERROR rule on HDFS: a window is anomalous if
one of its lines has level FATAL, FAILURE, SEVERE or ERROR.

Input: BGL.log from Loghub (https://github.com/logpai/loghub, about 4.7M lines).

Usage:
    python src/detect_bgl.py --log path/to/BGL.log
    python src/detect_bgl.py --log path/to/BGL.log --window 30
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict

import numpy as np

from detect import ROOT, CountVectorizer, LogRegDetector, PCADetector, prf

OUT_DIR = os.path.join(ROOT, "results", "detection_bgl")
BAD_LEVELS = {"FATAL", "FAILURE", "SEVERE", "ERROR"}


def bgl_template(component, level, msg):
    msg = re.sub(r"0x[0-9a-fA-F]+", "<HEX>", msg)
    msg = re.sub(r"\d+\.\d+\.\d+\.\d+(:\d+)?", "<IP>", msg)
    msg = re.sub(r"/[\w./\-]+", "<PATH>", msg)
    msg = re.sub(r"\b[0-9a-fA-F]{8,}\b", "<HEX>", msg)
    msg = re.sub(r"\d+", "<N>", msg)
    return f"{component} {level} {msg.strip()}"


def read_windows(path, minutes, max_lines=None):
    """Returns the windows in time order: (templates, any alert line, any bad level)."""
    windows = defaultdict(lambda: [[], 0, 0])
    n = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if max_lines and n >= max_lines:
                break
            parts = line.rstrip("\n").split(" ", 9)
            if len(parts) < 10 or not parts[1].isdigit():
                continue
            n += 1
            label, ts, component, level, msg = parts[0], int(parts[1]), parts[7], parts[8], parts[9]
            w = windows[ts // (60 * minutes)]
            w[0].append(bgl_template(component, level, msg))
            w[1] |= label != "-"
            w[2] |= level in BAD_LEVELS
    print(f"read {n:,} lines, {len(windows):,} windows of {minutes} min")
    return [windows[k] for k in sorted(windows)], n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="BGL.log from Loghub")
    ap.add_argument("--window", type=int, default=60, help="window length in minutes")
    ap.add_argument("--max-lines", type=int, default=None)
    ap.add_argument("--train-frac", type=float, default=0.8)
    args = ap.parse_args()

    t0 = time.perf_counter()
    windows, n_lines = read_windows(args.log, args.window, args.max_lines)
    t_parse = time.perf_counter() - t0
    y = np.array([w[1] for w in windows], dtype=int)
    cut = int(len(windows) * args.train_frac)
    tr, te = windows[:cut], windows[cut:]
    y_tr, y_te = y[:cut], y[cut:]
    print(f"train: {len(tr):,} windows ({y_tr.mean():.1%} anomalous), "
          f"test: {len(te):,} windows ({y_te.mean():.1%} anomalous)")

    vec = CountVectorizer().fit([w[0] for w in tr])
    X_tr = vec.transform([w[0] for w in tr])
    X_te = vec.transform([w[0] for w in te])
    unk = int((X_te[:, vec.unk] > 0).sum())
    print(f"{len(vec.names) - 1} templates in training, {unk} test windows contain an unseen template")

    results = {"Level rule (FATAL/FAILURE/SEVERE/ERROR)": prf(y_te, np.array([w[2] for w in te], dtype=int))}
    t1 = time.perf_counter()
    results["PCA (unsupervised)"] = prf(y_te, PCADetector().fit(X_tr).predict(X_te))
    results["PCA (unsupervised)"]["fit_s"] = round(time.perf_counter() - t1, 2)
    t1 = time.perf_counter()
    lr = LogRegDetector().fit(X_tr, y_tr)
    results["Logistic regression"] = prf(y_te, lr.predict(X_te))
    results["Logistic regression"]["fit_s"] = round(time.perf_counter() - t1, 2)

    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {"lines": n_lines, "window_minutes": args.window, "windows": len(windows),
               "train_windows": len(tr), "test_windows": len(te),
               "test_anomaly_rate": round(float(y_te.mean()), 4),
               "templates": len(vec.names) - 1, "test_windows_with_unseen_template": unk,
               "parse_s": round(t_parse, 1), "detectors": results}
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md = [f"# BGL detection ({n_lines:,} lines, {len(windows):,} windows of {args.window} min)", "",
          f"Chronological split, test = last {len(te):,} windows ({y_te.mean():.1%} anomalous). "
          f"{len(vec.names) - 1} templates in training, {unk} test windows contain an unseen one.", "",
          "| Detector | Precision | Recall | F1 | Flagged |",
          "|---|---:|---:|---:|---:|"]
    for name, r in results.items():
        md.append(f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['flagged']} |")
    with open(os.path.join(OUT_DIR, "metrics.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
