"""
Do the explanations point at what is actually wrong with the block?

ROUGE and BERTScore only compare the fine-tuned model with the teacher. Here I check both
against the raw log instead. For every anomalous test line, the script rebuilds the whole
block from HDFS.log and finds the block-level evidence:
  - the events that almost every normal block has (90%+ of normal training blocks) are
    missing, including the addStoredBlock that confirms the block was stored: the write
    never completed
  - otherwise the rare events it contains (seen in less than 5% of normal blocks), such
    as a failed delete or an addStoredBlock for a block that belongs to no file

Then it checks whether the cause and the reasoning written by the teacher and by the
fine-tuned model mention that evidence (keywords, see EVIDENCE below). I also read 50 of
them by hand (25 anomalies, 25 normal lines, seed 0) to check that the keyword match
does not miss anything; the sample is saved with the results.

Needs the detector trained by src/detect.py (it stores how often each template appears
in normal blocks) and the predictions in results/eval_n527/.

Usage:
    python src/audit_explanations.py --log path/to/HDFS.log
"""

import argparse
import json
import os
import pickle
import random
import re
from collections import Counter

from detect import BLK_RE, MODEL_PATH, ROOT
from evaluate import TEST_JSON, load, template

PREDS = os.path.join(ROOT, "results", "eval_n527", "preds_finetuned.jsonl")
OUT_DIR = os.path.join(ROOT, "results", "audit")

# evidence type -> (how to recognise it in the block, words an explanation would use)
EVIDENCE = {
    "write never completed": (None, r"incompl|inachev|jamais|interromp|manqu|absen|non termin|pas termin"),
    "failed delete (BlockInfo not found)": ("Unexpected error trying to delete", r"suppr|delet|effac|volumeMap|BlockInfo"),
    "addStoredBlock for a block of no file": ("addStoredBlock request received", r"addStoredBlock|redondant|redundant|appartient|orphelin|aucun fichier"),
    "empty packet": ("empty packet", r"paquet vide|empty packet"),
    "extra replication": ("replicate", r"r[ée]plica|replicat|transf[ée]r.*autre"),
}


def block_evidence(templates, normal_freq):
    present = set(templates)
    missing = [t for t, f in normal_freq.items() if f >= 0.9 and t not in present]
    if any("addStoredBlock: blockMap updated" in t for t in missing):
        return "write never completed", missing
    rare = [t for t in present if normal_freq.get(t, 0) < 0.05]
    for name, (needle, _) in EVIDENCE.items():
        if needle and any(needle in t for t in rare):
            return name, rare
    return "other", rare


def parse(raw):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="HDFS.log from Loghub HDFS_v1")
    args = ap.parse_args()

    with open(MODEL_PATH, "rb") as f:
        normal_freq = pickle.load(f)["normal_freq"]
    test = load(TEST_JSON)
    with open(PREDS, encoding="utf-8") as f:
        preds = [json.loads(l) for l in f]

    blocks = {e["block_id"]: [] for e in test}
    with open(args.log, encoding="utf-8", errors="ignore") as f:
        for line in f:
            for b in set(BLK_RE.findall(line)):
                if b in blocks:
                    blocks[b].append(template(line.strip()))

    rows = []
    for e, p in zip(test, preds):
        assert e["block_id"] == p["block_id"]
        ev, detail = block_evidence(blocks[e["block_id"]], normal_freq)
        q = parse(p["raw"])
        row = {"block_id": e["block_id"], "label": e["label"], "log": e["log"],
               "block_lines": len(blocks[e["block_id"]]), "evidence": ev, "evidence_detail": detail,
               "teacher": {"cause": e["cause"], "raisonnement": e["raisonnement"]},
               "finetuned": {"cause": q.get("cause"), "raisonnement": q.get("raisonnement")}}
        if e["label"] == "Anomaly" and ev in EVIDENCE:
            pat = re.compile(EVIDENCE[ev][1], re.I)
            for who in ("teacher", "finetuned"):
                text = f"{row[who]['cause'] or ''} {row[who]['raisonnement'] or ''}"
                row[who]["mentions_evidence"] = bool(pat.search(text))
        rows.append(row)

    anomalies = [r for r in rows if r["label"] == "Anomaly"]
    ev_count = Counter(r["evidence"] for r in anomalies)
    hits = {who: sum(r[who].get("mentions_evidence", False) for r in anomalies)
            for who in ("teacher", "finetuned")}
    causes = {who: Counter(r[who]["cause"] for r in anomalies).most_common(5)
              for who in ("teacher", "finetuned")}

    random.seed(0)
    sample = (random.sample(anomalies, 25)
              + random.sample([r for r in rows if r["label"] == "Normal"], 25))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "audit.json"), "w", encoding="utf-8") as f:
        json.dump({"evidence": ev_count, "mentions_evidence": hits, "top_causes": causes,
                   "n_anomalies": len(anomalies)}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "manual_sample.json"), "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=1)

    n = len(anomalies)
    md = [f"# Explanation audit ({n} anomalous test lines)", "",
          "## What is actually wrong with the block", "",
          "| Block-level evidence | Blocks |", "|---|---:|"]
    md += [f"| {k} | {v} ({v / n:.0%}) |" for k, v in ev_count.most_common()]
    md += ["", "## Do the explanations mention it?", "",
           "| System | Mentions the block-level evidence |", "|---|---:|"]
    md += [f"| {who} | {hits[who]} / {n} |" for who in ("teacher", "finetuned")]
    md += ["", "## Most frequent causes given", ""]
    for who in ("teacher", "finetuned"):
        md.append(f"- {who}: " + "; ".join(f"{c} ({k})" for c, k in causes[who]))
    with open(os.path.join(OUT_DIR, "audit.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
