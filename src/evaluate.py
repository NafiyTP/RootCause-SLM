"""
Evaluate the fine-tuned model on the held-out test set (data/hdfs_test_dataset.json).

The model is an annotator: given a log line and its label, it writes a cause and a
3-step reasoning. The references were written by Llama 3.3-70B, so every score here
measures agreement with the teacher, not ground-truth correctness.

Systems compared:
  - most frequent cause from the training set
  - majority cause per label
  - template retrieval: copy the annotation of a training line with the same
    message template and label (strongest non-LLM baseline)
  - Qwen2.5-1.5B-Instruct zero-shot
  - Qwen2.5-1.5B-Instruct few-shot (3 fixed training examples)
  - Qwen2.5-1.5B-Instruct + my LoRA adapter

Metrics: JSON validity (strict and lenient), cause exact match, ROUGE-L, BERTScore,
split by Normal/Anomaly, plus bootstrap confidence intervals.

Every generation is cached in results/eval_n{N}/preds_*.jsonl, so an interrupted run
picks up where it stopped. Each value of --n gets its own folder: an earlier version
of this script shared one results file between the 20-example check and the full run,
and the full run silently reused the 20-example numbers.

Usage:
    python src/evaluate.py                 # full test set (527 lines)
    python src/evaluate.py --n 20          # quick check
    python src/evaluate.py --no-llm        # baselines only, runs on CPU
    python src/evaluate.py --no-bertscore
"""

import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np

SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.join(SRC_DIR, "..")
TEST_JSON  = os.path.join(ROOT, "data", "hdfs_test_dataset.json")
TRAIN_JSON = os.path.join(ROOT, "data", "hdfs_dataset.json")
LORA_DIR   = os.path.join(ROOT, "modele_hdfs")

# Same system prompt as src/dataset.py. The data is in French, so the prompts are too.
SYSTEM_PROMPT = (
    "Tu es un expert en systèmes distribués Hadoop/HDFS. "
    "Étant donné un log et son label (Normal ou Anomaly), "
    "tu fournis une cause technique précise et un raisonnement en 3 étapes."
)
# The base model needs to be told the output format; the fine-tuned one learned it.
JSON_INSTR = (
    "\n\nRéponds en JSON pur sans backticks :\n"
    '{"cause": "une phrase courte", '
    '"raisonnement": "Étape 1 : ... Étape 2 : ... Étape 3 : ..."}'
)


def user_prompt(entry):
    return f"Log HDFS : {entry['log'][:300]}\nLabel : {entry['label']}"


def target_json(entry):
    return json.dumps({"cause": entry["cause"], "raisonnement": entry["raisonnement"]},
                      ensure_ascii=False)


# ---------------------------------------------------------------- data

def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [d for d in data if d.get("cause") and d.get("raisonnement")]


def template(log):
    """Message skeleton: drop date/time/pid, then mask block ids, IPs, paths and numbers."""
    parts = log.split(" ", 3)
    msg = parts[3] if len(parts) == 4 else log
    msg = re.sub(r"blk_-?\d+", "<BLK>", msg)
    msg = re.sub(r"\d+\.\d+\.\d+\.\d+(:\d+)?", "<IP>", msg)
    msg = re.sub(r"/[\w./\-]+", "<PATH>", msg)
    msg = re.sub(r"\d+", "<N>", msg)
    return msg.strip()


def norm(s):
    s = unicodedata.normalize("NFKC", s or "").lower().strip()
    return re.sub(r"[\s\.\!]+$", "", re.sub(r"\s+", " ", s))


# ---------------------------------------------------------------- non-LLM baselines

def baseline_most_frequent(train, test):
    cause = Counter(d["cause"] for d in train).most_common(1)[0][0]
    reason = next(d["raisonnement"] for d in train if d["cause"] == cause)
    return [{"ok": True, "cause": cause, "raisonnement": reason} for _ in test]


def baseline_label_majority(train, test):
    by_label = defaultdict(Counter)
    for d in train:
        by_label[d["label"]][d["cause"]] += 1
    out = []
    for e in test:
        cause = by_label[e["label"]].most_common(1)[0][0]
        reason = next(d["raisonnement"] for d in train if d["cause"] == cause)
        out.append({"ok": True, "cause": cause, "raisonnement": reason})
    return out


def baseline_retrieval(train, test):
    """Nearest neighbour by template, same label when possible, else label majority."""
    index = defaultdict(list)
    for d in train:
        index[(template(d["log"]), d["label"])].append(d)
        index[(template(d["log"]), None)].append(d)
    fallback = baseline_label_majority(train, test)
    out, hits = [], Counter()
    for e, fb in zip(test, fallback):
        t = template(e["log"])
        cands = index.get((t, e["label"])) or index.get((t, None))
        if cands:
            hits["template+label" if index.get((t, e["label"])) else "template"] += 1
            cause = Counter(c["cause"] for c in cands).most_common(1)[0][0]
            reason = next(c["raisonnement"] for c in cands if c["cause"] == cause)
            out.append({"ok": True, "cause": cause, "raisonnement": reason})
        else:
            hits["fallback"] += 1
            out.append(fb)
    print(f"  retrieval matches: {dict(hits)}")
    return out


# ---------------------------------------------------------------- parsing

def parse(text):
    """Returns (strict_ok, lenient_ok, dict). Lenient = first {...} block in the text."""
    t = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    t = re.sub(r"```$", "", t).strip()
    try:
        d = json.loads(t)
        if isinstance(d, dict):
            return True, True, d
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict):
                return False, True, d
        except json.JSONDecodeError:
            pass
    return False, False, {}


def to_pred(text):
    strict, lenient, d = parse(text)
    cause = d.get("cause", "") if lenient else ""
    reason = d.get("raisonnement", d.get("reasoning", "")) if lenient else ""
    if isinstance(reason, (list, dict)):
        reason = json.dumps(reason, ensure_ascii=False)
    return {"ok": strict, "ok_lenient": lenient, "cause": str(cause or ""),
            "raisonnement": str(reason or ""), "raw": text}


# ---------------------------------------------------------------- generation

def build_messages(entry, mode, shots):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if mode == "finetuned":  # exactly the training format
        msgs.append({"role": "user", "content": user_prompt(entry)})
        return msgs
    if mode == "fewshot":
        for s in shots:
            msgs.append({"role": "user", "content": user_prompt(s) + JSON_INSTR})
            msgs.append({"role": "assistant", "content": target_json(s)})
    msgs.append({"role": "user", "content": user_prompt(entry) + JSON_INSTR})
    return msgs


def pick_shots(train):
    """One Normal line with the most frequent cause, then two Anomaly lines with different causes."""
    top = Counter(d["cause"] for d in train).most_common(1)[0][0]
    shots = [next(d for d in train if d["cause"] == top)]
    seen = set()
    for d in train:
        if d["label"] == "Anomaly" and d["cause"] not in seen:
            shots.append(d)
            seen.add(d["cause"])
        if len(shots) == 3:
            break
    return shots


def generate_all(test, mode, base_model, lora_dir, shots, cache_path, batch_size, max_new):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                cache[r["block_id"]] = r["raw"]
    todo = [e for e in test if e["block_id"] not in cache]
    print(f"  {len(cache)} cached, {len(todo)} to generate")

    if todo:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(base_model)
        tok.padding_side = "left"  # left padding for batched generation
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        if mode == "finetuned":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_dir)
        model.eval()
        stop_ids = [i for i in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                                tok.convert_tokens_to_ids("<|endoftext|>")}
                    if isinstance(i, int) and i >= 0]

        with open(cache_path, "a", encoding="utf-8") as f:
            for b in range(0, len(todo), batch_size):
                batch = todo[b:b + batch_size]
                prompts = [tok.apply_chat_template(build_messages(e, mode, shots),
                                                   tokenize=False, add_generation_prompt=True)
                           for e in batch]
                enc = tok(prompts, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                         pad_token_id=tok.pad_token_id, eos_token_id=stop_ids)
                gen = out[:, enc["input_ids"].shape[1]:]
                for e, g in zip(batch, gen):
                    raw = tok.decode(g, skip_special_tokens=True).strip()
                    cache[e["block_id"]] = raw
                    f.write(json.dumps({"block_id": e["block_id"], "raw": raw},
                                       ensure_ascii=False) + "\n")
                f.flush()
                print(f"  {min(b + batch_size, len(todo))}/{len(todo)}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    return [to_pred(cache[e["block_id"]]) for e in test]


# ---------------------------------------------------------------- metrics

def rouge_per_example(preds, refs):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return np.array([sc.score(r, p)["rougeL"].fmeasure if p else 0.0
                     for p, r in zip(preds, refs)])


def bert_per_example(preds, refs):
    import torch
    from bert_score import score
    _, _, f1 = score([p if p else "-" for p in preds], refs, lang="fr", verbose=False,
                     batch_size=64, device="cuda" if torch.cuda.is_available() else "cpu")
    f1 = f1.cpu().numpy()
    f1[[i for i, p in enumerate(preds) if not p]] = 0.0  # empty output scores 0
    return f1


def boot_ci(x, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compute_metrics(name, preds, test, use_bert):
    labels = np.array([e["label"] for e in test])
    per = {
        "json_strict":  np.array([float(p.get("ok", False)) for p in preds]),
        "json_lenient": np.array([float(p.get("ok_lenient", p.get("ok", False))) for p in preds]),
        "cause_exact":  np.array([float(norm(p["cause"]) == norm(e["cause"]))
                                  for p, e in zip(preds, test)]),
        "rouge_cause":  rouge_per_example([p["cause"] for p in preds], [e["cause"] for e in test]),
        "rouge_reason": rouge_per_example([p["raisonnement"] for p in preds],
                                          [e["raisonnement"] for e in test]),
    }
    if use_bert:
        per["bert_cause"] = bert_per_example([p["cause"] for p in preds], [e["cause"] for e in test])
        per["bert_reason"] = bert_per_example([p["raisonnement"] for p in preds],
                                              [e["raisonnement"] for e in test])
    res = {"name": name, "n": len(test)}
    for k, v in per.items():
        res[k] = round(float(v.mean()), 4)
        for lab in ("Normal", "Anomaly"):
            m = labels == lab
            if m.any():
                res[f"{k}__{lab}"] = round(float(v[m].mean()), 4)
    for k in ("rouge_reason", "rouge_cause"):
        lo, hi = boot_ci(per[k])
        res[f"{k}_ci95"] = [round(lo, 4), round(hi, 4)]
    return res


# ---------------------------------------------------------------- reports

def write_reports(out_dir, results, all_preds, test):
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    cols = ["json_strict", "json_lenient", "cause_exact", "rouge_cause", "rouge_reason"]
    if "bert_cause" in results[0]:
        cols += ["bert_cause", "bert_reason"]
    lines = [f"# Results (n = {results[0]['n']})", "",
             "| System | " + " | ".join(cols) + " |",
             "|---|" + "---:|" * len(cols)]
    for r in results:
        lines.append(f"| {r['name']} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    lines += ["", "## By label", "",
              "| System | ROUGE-L reasoning, Normal | ROUGE-L reasoning, Anomaly "
              "| cause exact match, Normal | cause exact match, Anomaly |",
              "|---|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['name']} | {r.get('rouge_reason__Normal', 0):.3f} | "
                     f"{r.get('rouge_reason__Anomaly', 0):.3f} | {r.get('cause_exact__Normal', 0):.3f} | "
                     f"{r.get('cause_exact__Anomaly', 0):.3f} |")
    lines += ["", "## ROUGE-L reasoning, bootstrap 95% CI", ""]
    for r in results:
        lo, hi = r["rouge_reason_ci95"]
        lines.append(f"- {r['name']}: {r['rouge_reason']:.3f} [{lo:.3f}, {hi:.3f}]")
    with open(os.path.join(out_dir, "metrics.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # A few test lines side by side: the first 3 Normal and the first 5 Anomaly
    idx = [i for i, e in enumerate(test) if e["label"] == "Normal"][:3] + \
          [i for i, e in enumerate(test) if e["label"] == "Anomaly"][:5]
    ex = ["# Example outputs", ""]
    for i in idx:
        e = test[i]
        ex += [f"## [{e['label']}] {e['log'][:160]}", "",
               f"**Reference (Llama 3.3-70B)**, cause: {e['cause']}", "",
               f"> {e['raisonnement']}", ""]
        for name, preds in all_preds.items():
            p = preds[i]
            shown = p["raisonnement"] or (p.get("raw", "")[:300] + " [invalid JSON]")
            ex += [f"**{name}**, cause: {p['cause'] or '(none)'}", "", f"> {shown}", ""]
        ex += ["---", ""]
    with open(os.path.join(out_dir, "examples.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(ex))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="evaluate on the first n test lines")
    ap.add_argument("--no-llm", action="store_true", help="baselines only")
    ap.add_argument("--no-bertscore", action="store_true")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--lora-dir", default=LORA_DIR)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    train, test = load(TRAIN_JSON), load(TEST_JSON)
    if args.n:
        test = test[:args.n]
    out_dir = args.out or os.path.join(ROOT, "results", f"eval_n{len(test)}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"train={len(train)} test={len(test)} {dict(Counter(e['label'] for e in test))}")
    print(f"writing to {out_dir}")

    systems = {
        "Most frequent cause": baseline_most_frequent(train, test),
        "Label majority": baseline_label_majority(train, test),
        "Template retrieval (kNN)": baseline_retrieval(train, test),
    }
    if not args.no_llm:
        shots = pick_shots(train)
        for name, mode in [("Qwen2.5-1.5B zero-shot", "zeroshot"),
                           ("Qwen2.5-1.5B few-shot (3)", "fewshot"),
                           ("Qwen2.5-1.5B + LoRA (ours)", "finetuned")]:
            print(f"\n{name}")
            cache = os.path.join(out_dir, f"preds_{mode}.jsonl")
            systems[name] = generate_all(test, mode, args.base_model, args.lora_dir, shots,
                                         cache, args.batch_size, args.max_new_tokens)

    results = []
    for name, preds in systems.items():
        print(f"scoring {name}")
        results.append(compute_metrics(name, preds, test, not args.no_bertscore))
    write_reports(out_dir, results, systems, test)
    with open(os.path.join(out_dir, "metrics.md"), encoding="utf-8") as f:
        print("\n" + f.read())


if __name__ == "__main__":
    main()
