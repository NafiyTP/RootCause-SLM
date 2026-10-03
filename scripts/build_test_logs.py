"""
Build the raw test log file: one line per HDFS block that is not in the training set,
250 anomalous blocks and 250 normal ones.

It streams the full HDFS_v1 log (HDFS.log, about 1.5 GB, from Loghub) and keeps the first
line it meets for each new labeled block. Taking the first line means most anomalous blocks
are represented by their allocateBlock line, which is not anomalous by itself (see the
Limitations section of the README).

Usage:
    python scripts/build_test_logs.py --hdfs-log raw_data/HDFS.log \
        --labels raw_data/anomaly_label.csv --out raw_data/HDFS_test_500.log
"""

import argparse
import json
import random
import re

BLOCK_RE = re.compile(r"blk_-?\d+")


def load_labels(path):
    labels = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # header: BlockId,Label
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2:
                labels[parts[0].strip()] = parts[1].strip()
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdfs-log", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--train-json", default="data/hdfs_dataset.json")
    ap.add_argument("--out", default="raw_data/HDFS_test_500.log")
    ap.add_argument("--per-class", type=int, default=250)
    args = ap.parse_args()

    with open(args.train_json, encoding="utf-8") as f:
        used = {d["block_id"] for d in json.load(f) if "block_id" in d}
    print(f"{len(used)} training blocks excluded")

    labels = load_labels(args.labels)
    anomalies, normals = [], []

    with open(args.hdfs_log, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i and i % 500_000 == 0:
                print(f"{i} lines read, {len(anomalies)} anomalies, {len(normals)} normal")
            m = BLOCK_RE.search(line)
            if not m:
                continue
            block = m.group(0)
            if block in used or block not in labels:
                continue
            if labels[block] == "Anomaly" and len(anomalies) < args.per_class:
                anomalies.append(line.strip())
                used.add(block)
            elif labels[block] == "Normal" and len(normals) < args.per_class:
                normals.append(line.strip())
                used.add(block)
            if len(anomalies) >= args.per_class and len(normals) >= args.per_class:
                break

    lines = anomalies + normals
    random.shuffle(lines)  # not seeded in the original run
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} lines ({len(anomalies)} Anomaly, {len(normals)} Normal) to {args.out}")


if __name__ == "__main__":
    main()
