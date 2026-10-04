"""
Builds the v2 explanation dataset: one example per block, with the block summary as input.

Same chronological split as src/detect.py (blocks sorted by first appearance, first 80%
for training), so training blocks come from the past and test blocks from the future.
Anomalous blocks are sampled per gold category (so rare kinds of failure are present),
normal blocks are half random and half "re-replication" blocks, the pattern behind all of
the detector's false alarms, to check whether the model invents a problem when there is none.

Needs the detector trained by src/detect.py (for the normal-event frequencies).

Usage:
    python src/build_dataset_v2.py --log path/to/HDFS.log --labels path/to/anomaly_label.csv
Writes data/v2/train_blocks.json and data/v2/test_blocks.json (not annotated yet).
"""

import argparse
import json
import os
import pickle
import random
from collections import defaultdict

from blocks_v2 import block_prompt, gold_category
from detect import BLK_RE, MODEL_PATH, ROOT, read_labels
from evaluate import template
from pipeline import load_detector, missing_events, pick_line, rare_events

OUT_DIR = os.path.join(ROOT, "data", "v2")

# per-category quota for anomalous blocks, train / test
QUOTA = {"train": 45, "test": 15}
N_NORMAL = {"train": 100, "test": 40}


def read_blocks(path):
    blocks = defaultdict(list)
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            for b in set(BLK_RE.findall(line)):
                blocks[b].append((template(line), line))
    return blocks


def sample(ids, blocks, labels, normal_freq, split, rng, vec, coefs):
    by_cat = defaultdict(list)
    normals, rerepl = [], []
    rule_cat = {}
    for b in ids:
        lines = blocks[b]
        cat = gold_category(missing_events(lines, normal_freq), rare_events(lines, normal_freq))
        rule_cat[b] = cat
        if labels[b]:
            by_cat[cat].append(b)
        elif cat == "re_replication":
            rerepl.append(b)
        else:
            normals.append(b)
    chosen = []
    for cat, bs in sorted(by_cat.items()):
        rng.shuffle(bs)
        chosen += [(b, "Anomaly", cat) for b in bs[:QUOTA[split]]]
    n = N_NORMAL[split]
    rng.shuffle(rerepl)
    rng.shuffle(normals)
    k = min(n // 2, len(rerepl))
    chosen += [(b, "Normal", "re_replication") for b in rerepl[:k]]
    chosen += [(b, "Normal", "none") for b in normals[:n - k]]
    rng.shuffle(chosen)
    print(f"{split}: {len(chosen)} blocks, anomalous per category: "
          + ", ".join(f"{c} {min(len(v), QUOTA[split])}/{len(v)}" for c, v in sorted(by_cat.items()))
          + f"; normal {n - k} random + {k} re-replication")
    out = []
    for b, label, cat in chosen:
        lines = blocks[b]
        out.append({"block_id": b, "label": label, "category": cat, "n_lines": len(lines),
                    # what the pipeline rules say about this block (for normal blocks too)
                    "rule_category": rule_cat[b],
                    "prompt": block_prompt(b, lines, normal_freq),
                    # the single line v1 would have received, to compare v1 and v2 on the same blocks
                    "v1_line": pick_line(lines, vec, coefs)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(MODEL_PATH, "rb") as f:
        normal_freq = pickle.load(f)["normal_freq"]
    vec, coefs, _, _ = load_detector()
    labels = read_labels(args.labels)
    blocks = read_blocks(args.log)
    ids = [b for b in blocks if b in labels]          # order of first appearance
    cut = int(len(ids) * args.train_frac)
    rng = random.Random(args.seed)

    os.makedirs(OUT_DIR, exist_ok=True)
    for split, part in (("train", ids[:cut]), ("test", ids[cut:])):
        data = sample(part, blocks, labels, normal_freq, split, rng, vec, coefs)
        lengths = sorted(len(d["prompt"]) for d in data)
        print(f"  prompt length (chars): median {lengths[len(lengths) // 2]}, max {lengths[-1]}")
        with open(os.path.join(OUT_DIR, f"{split}_blocks.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
