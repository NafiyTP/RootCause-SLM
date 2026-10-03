"""
Annotate HDFS log lines with Llama 3.3-70B through the Groq API.

For each line, the teacher gets the log and its block label and returns
{"cause": ..., "raisonnement": ...} in French. The label always comes from Loghub;
the LLM only writes the explanation.

I used it twice:
  - training set: the 2,000-line Loghub sample HDFS_2k.log -> data/hdfs_dataset.json
  - test set: the file built by build_test_logs.py -> data/hdfs_test_dataset.json

The output is saved after every line and already annotated blocks are skipped on restart,
so it can be stopped and resumed (the free Groq tier runs out of quota quickly).

Usage:
    export GROQ_API_KEY=...
    python scripts/annotate_with_llm.py \
        --logs https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log \
        --labels https://raw.githubusercontent.com/logpai/loglizer/master/data/HDFS/anomaly_label.csv \
        --out data/hdfs_dataset
    python scripts/annotate_with_llm.py --logs raw_data/HDFS_test_500.log \
        --labels raw_data/anomaly_label.csv --out data/hdfs_test_dataset
"""

import argparse
import csv
import json
import os
import re
import time
from collections import Counter

import requests

MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
PAUSE_SEC = 4.0          # between calls, to stay under the rate limit
MAX_CONSECUTIVE_429 = 5  # then stop and resume later
BLOCK_RE = re.compile(r"blk_-?\d+")

PROMPT = """Log HDFS : {log}
Label : {label}

Réponds en JSON pur sans backticks :
{{"cause": "une phrase courte et précise", "raisonnement": "Étape 1 : ... Étape 2 : ... Étape 3 : ..."}}"""


def read_text(path_or_url):
    if path_or_url.startswith("http"):
        return requests.get(path_or_url, timeout=30).text
    with open(path_or_url, encoding="utf-8") as f:
        return f.read()


def load_labels(path_or_url):
    labels = {}
    for line in read_text(path_or_url).strip().split("\n")[1:]:
        parts = line.split(",")
        if len(parts) == 2:
            labels[parts[0].strip()] = parts[1].strip()
    return labels


def annotate(log, label, api_key):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system",
             "content": "Expert Hadoop/HDFS. JSON pur uniquement, sans backticks. Réponses courtes."},
            {"role": "user", "content": PROMPT.format(log=log[:200], label=label)},
        ],
        "temperature": 0.2,
        "max_tokens": 200,
    }
    resp = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {api_key}"},
                         json=payload, timeout=30)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    content = re.sub(r"^`{3}(?:json)?", "", content).strip().rstrip("`").strip()
    return json.loads(content)


def save(rows, out):
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(out + ".csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["block_id", "log", "label", "cause", "raisonnement"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="log file path or URL")
    ap.add_argument("--labels", required=True, help="anomaly_label.csv path or URL")
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--limit", type=int, default=None, help="annotate only the first N lines")
    args = ap.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("set GROQ_API_KEY first")

    rows = []
    if os.path.exists(args.out + ".json"):
        with open(args.out + ".json", encoding="utf-8") as f:
            rows = json.load(f)
    done = {r["block_id"] for r in rows}

    labels = load_labels(args.labels)
    lines = []
    for line in read_text(args.logs).strip().split("\n"):
        m = BLOCK_RE.search(line)
        if m and m.group(0) in labels:
            lines.append({"line": line.strip(), "block_id": m.group(0), "label": labels[m.group(0)]})
    if args.limit:
        lines = lines[:args.limit]
    todo = [e for e in lines if e["block_id"] not in done]
    print(f"{len(lines)} labeled lines, {len(done)} already done, {len(todo)} to annotate")

    n_429 = 0
    i = 0
    while i < len(todo):
        e = todo[i]
        row = {"block_id": e["block_id"], "log": e["line"][:300], "label": e["label"],
               "cause": "", "raisonnement": ""}
        try:
            ann = annotate(e["line"], e["label"], api_key)
            row["cause"] = ann.get("cause", "")
            row["raisonnement"] = ann.get("raisonnement", "")
            n_429 = 0
        except json.JSONDecodeError:
            pass  # keep the line with empty fields; dataset.py skips it
        except requests.HTTPError as err:
            code = err.response.status_code if err.response is not None else 0
            if code != 429:
                print(f"HTTP {code}: {err}")
                break
            n_429 += 1
            if n_429 >= MAX_CONSECUTIVE_429:
                print("out of quota, run again later to resume")
                break
            time.sleep(min(10 * n_429, 60))
            continue  # retry the same line
        rows.append(row)
        save(rows, args.out)
        print(f"[{len(rows)}] {e['label']:7s} | {row['cause'][:60]}")
        i += 1
        time.sleep(PAUSE_SEC)

    save(rows, args.out)
    print(f"done: {dict(Counter(r['label'] for r in rows))}")


if __name__ == "__main__":
    main()
