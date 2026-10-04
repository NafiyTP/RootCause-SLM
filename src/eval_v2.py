"""
Evaluates the explanations on the v2 test blocks, against the raw log, not the teacher.

Systems compared on the same 123 test blocks (83 anomalous, 40 normal):
  rules     the pipeline's rule-based sentence (what we use today)
  teacher   the large model's annotation (data/v2/test_annotated.json)
  v1        the v1 student: one line + the label "Anomaly", French answer (modele_hdfs/)
  base      Qwen2.5-1.5B-Instruct without fine-tuning, given the v2 block summary
  v2        the v2 student: block summary in, JSON out (modele_hdfs_v2/)

Metrics (see blocks_v2.score):
  correct cause   anomalous blocks whose explanation names the right kind of failure
  wrong cause     anomalous blocks whose explanation names another kind of failure
  invents         normal blocks for which the system claims a problem (random normal
                  blocks, and re-replication blocks, the pattern behind the detector's
                  false alarms)
  evidence        share of cited evidence that is actually in the input

v1 always gets the label "Anomaly", so its normal-block columns are left empty.

Usage:
    python src/eval_v2.py                         # all systems (GPU for v1/base/v2)
    python src/eval_v2.py --systems rules teacher # CPU only
Writes results/v2/metrics.md, metrics.json, preds_<system>.jsonl, examples.md.
"""

import argparse
import json
import os

from blocks_v2 import CATEGORIES, SYSTEM_PROMPT, parse_answer, score
from detect import ROOT
from pipeline import RULES

DATA = os.path.join(ROOT, "data", "v2")
OUT = os.path.join(ROOT, "results", "v2")
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
RULE_SENTENCE = dict(zip(CATEGORIES, [s for _, s in RULES]))


def load_test():
    path = os.path.join(DATA, "test_annotated.json")
    if not os.path.exists(path):
        path = os.path.join(DATA, "test_blocks.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rules_answers(test):
    out = []
    for x in test:
        c = x["rule_category"]
        if c == "none":
            out.append({"anomalous": False, "cause": "", "evidence": [], "explanation": ""})
        else:
            out.append({"anomalous": True, "cause": RULE_SENTENCE[c], "evidence": [], "explanation": ""})
    return out


def generate(model, tok, messages_list, max_new=300, batch_size=8):
    import torch
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    stop = [i for i in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
            if isinstance(i, int) and i >= 0]
    texts = []
    for i in range(0, len(messages_list), batch_size):
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in messages_list[i:i + batch_size]]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=stop)
        texts += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  {min(i + batch_size, len(messages_list))}/{len(messages_list)}")
    return texts


def load_model(adapter):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16).to("cuda")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return model, tok


def model_answers(system, test):
    if system == "v1":
        from dataset import SYSTEM_PROMPT as V1_SYSTEM, format_example
        model, tok = load_model(os.path.join(ROOT, "modele_hdfs"))
        msgs = [[{"role": "system", "content": V1_SYSTEM},
                 {"role": "user", "content": format_example(
                     {"log": x["v1_line"], "label": "Anomaly", "cause": "", "raisonnement": ""})[0]}]
                for x in test]
    else:
        adapter = os.path.join(ROOT, "modele_hdfs_v2") if system == "v2" else None
        model, tok = load_model(adapter)
        msgs = [[{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": x["prompt"]}] for x in test]
    raw = generate(model, tok, msgs)
    del model
    import torch
    torch.cuda.empty_cache()
    if system == "v1":
        return [parse_v1(t) for t in raw], raw
    return [parse_answer(t) for t in raw], raw


def parse_v1(text):
    """v1 answers {"cause", "raisonnement"} in French; it was told the block is anomalous."""
    t = (text or "").strip()
    try:
        j = json.loads(t[t.index("{"):t.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(j, dict) or not j.get("cause"):
        return None
    return {"anomalous": True, "cause": str(j["cause"]), "evidence": [],
            "explanation": str(j.get("raisonnement") or "")}


def fmt(v, pct=True):
    if v is None:
        return "-"
    return f"{100 * v:.0f}%" if pct else f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["rules", "teacher", "v1", "base", "v2"])
    ap.add_argument("--redo", action="store_true", help="ignore saved predictions")
    args = ap.parse_args()

    test = load_test()
    os.makedirs(OUT, exist_ok=True)
    results, answers_by = {}, {}
    for system in args.systems:
        path = os.path.join(OUT, f"preds_{system}.jsonl")
        if system == "rules":
            answers, raw = rules_answers(test), None
        elif system == "teacher":
            if "teacher" not in test[0]:
                print("no teacher annotations yet (data/v2/test_annotated.json), skipping")
                continue
            answers, raw = [x["teacher"] for x in test], None
        elif os.path.exists(path) and not args.redo:
            with open(path, encoding="utf-8") as f:
                saved = {r["block_id"]: r for r in map(json.loads, f)}
            answers = [saved[x["block_id"]]["answer"] for x in test]
            raw = [saved[x["block_id"]]["raw"] for x in test]
            print(f"{system}: loaded saved predictions")
        else:
            print(f"{system}: generating")
            answers, raw = model_answers(system, test)
        with open(path, "w", encoding="utf-8") as f:
            for x, a, r in zip(test, answers, raw or [None] * len(test)):
                f.write(json.dumps({"block_id": x["block_id"], "label": x["label"],
                                    "category": x["category"], "answer": a, "raw": r},
                                   ensure_ascii=False) + "\n")
        items = [dict(x, answer=a) for x, a in zip(test, answers)]
        res = score(items)
        if system == "v1":   # v1 is told "Anomaly": normal-block metrics are meaningless
            for k in ("invents_on_normal", "invents_on_normal_random", "invents_on_normal_rereplication"):
                res[k] = None
        results[system] = res
        answers_by[system] = answers

    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    names = {"rules": "Rules (pipeline sentence)", "teacher": "Teacher (large model)",
             "v1": "v1 student (one line)", "base": "Qwen2.5-1.5B, no fine-tuning",
             "v2": "v2 student (block summary)"}
    n_a = next(iter(results.values()))["n_anomaly"] if results else 0
    md = [f"# v2 explanations, {len(test)} test blocks ({n_a} anomalous, {len(test) - n_a} normal)", "",
          "Scored against the raw log (gold = what is wrong in the whole block), not against the teacher.", "",
          "| System | Valid JSON | Correct cause | Wrong cause | Invents on normal (random) | "
          "Invents on normal (re-replication) | Evidence in input |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for s, r in results.items():
        md.append(f"| {names.get(s, s)} | {fmt(r['valid_json'])} | {fmt(r['correct_cause'])} | "
                  f"{fmt(r['wrong_cause'])} | {fmt(r['invents_on_normal_random'])} | "
                  f"{fmt(r['invents_on_normal_rereplication'])} | {fmt(r['evidence_supported'])} |")
    md += ["", "Rules define the gold categories, so their correct-cause score is 100% by construction;",
           "the question is how close the language models get, and how often they invent.", "",
           "## Correct cause per category", "",
           "| Category | n | " + " | ".join(names.get(s, s) for s in results) + " |",
           "|---|---:|" + "---:|" * len(results)]
    for c in CATEGORIES:
        row = [results[s]["per_category"].get(c) for s in results]
        if any(row):
            n = next(r["n"] for r in row if r)
            md.append(f"| {c} | {n} | " + " | ".join(fmt(r["correct"]) if r else "-" for r in row) + " |")
    with open(os.path.join(OUT, "metrics.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    ex = ["# Examples", ""]
    seen = set()
    for i, x in enumerate(test):
        key = (x["label"], x["category"])
        if key in seen:
            continue
        seen.add(key)
        ex += [f"## {x['block_id']}: {x['label']}, {x['category']}", "", "```", x["prompt"], "```", ""]
        for s, ans in answers_by.items():
            a = ans[i]
            if a is None:
                ex.append(f"- **{s}**: (no valid answer)")
            else:
                ex.append(f"- **{s}**: anomalous={a['anomalous']}; {a['cause']}. {a['explanation']}")
        ex.append("")
    with open(os.path.join(OUT, "examples.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(ex) + "\n")


if __name__ == "__main__":
    main()
