"""
Shared pieces of the v2 explanation model (block-level input).

v1 gave the model one log line and its label, and the cause was not in that line.
v2 gives it the whole block, summarised: every event type with its count, in order of
first appearance, the raw WARN/exception lines, and the events a normal write always has.
The model never gets the ground-truth label, only "the detector flagged this block", so
it has to say itself whether something is wrong, and what.

The gold category of a block comes from the same rules as the pipeline sentence
(pipeline.describe), applied to the full block. For normal blocks the gold answer is
"nothing clearly wrong".
"""

import json
import re
from collections import Counter

from pipeline import RULES

# Events that (almost) every normal block has: >= 99% of normal training blocks.
USUAL_MIN_FREQ = 0.99

SYSTEM_PROMPT = (
    "You are an HDFS expert. You get a summary of one HDFS block: its events with counts, "
    "its WARN/exception lines, and the events a normal block always has. The block may or "
    "may not be anomalous. Say whether it shows a real problem and what it is, using only "
    "what is in the summary. Do not invent causes that the events do not show. "
    "Answer in JSON with the keys: anomalous (true/false), cause (a few words), "
    "evidence (list of events from the summary, prefix with 'missing: ' for an absent "
    "usual event), explanation (one or two sentences)."
)

# Gold categories, in the order of pipeline.RULES (the first rule that matches wins).
CATEGORIES = ["never_stored", "failed_delete", "no_file", "redundant", "empty_packet",
              "replication_timeout", "re_replication", "rare_other", "missing_other"]

# Words an explanation would use for each category (lower case, regex).
CATEGORY_WORDS = {
    "never_stored": r"never (been )?stored|not (been )?stored|never complet|not complet|incomplete|"
                    r"did not finish|never finish|unfinished|interrupted|missing addstoredblock|"
                    r"no addstoredblock|never (been )?(recorded|registered|confirmed)|write (failed|did not)|"
                    r"jamais (été )?(stock|enregistr|termin)|incomplet|inachev|interromp",
    "failed_delete": r"delet|suppr",
    "no_file": r"belong(s)? to no file|does not belong|not belong to any file|orphan|aucun fichier|n'appartient",
    "redundant": r"redundant|twice|duplicate|redondant|deux fois|doublon",
    "empty_packet": r"empty packet|paquet vide",
    "replication_timeout": r"timed out|timeout|time-out|expir",
    "re_replication": r"re-?replicat|replicat|copied to another|transfer(red)? to another|r[ée]plica",
}


def usual_events(normal_freq):
    return [t for t, f in sorted(normal_freq.items(), key=lambda kv: -kv[1]) if f >= USUAL_MIN_FREQ]


def gold_category(missing, rare):
    """Index of the first matching pipeline rule, as a category name."""
    for name, (test, _) in zip(CATEGORIES, RULES):
        if test(missing, rare):
            return name
    return "none"


def block_prompt(block_id, lines, normal_freq, max_raw=6):
    """lines: [(template, raw line)] in log order. Returns the user prompt text."""
    counts = Counter(t for t, _ in lines)
    order = list(dict.fromkeys(t for t, _ in lines))
    out = [f"Block {block_id}, {len(lines)} lines. Events in order of first appearance:"]
    out += [f"  {counts[t]}x {t}" for t in order]
    raw = [l for t, l in lines if " WARN " in l or "xception" in l]
    if raw:
        out.append("WARN or exception lines:")
        out += [f"  {l[:220]}" for l in raw[:max_raw]]
        if len(raw) > max_raw:
            out.append(f"  ... ({len(raw) - max_raw} more)")
    out.append("A normal block always has: " + "; ".join(usual_events(normal_freq)))
    return "\n".join(out)


def target_json(ann):
    return json.dumps({"anomalous": ann["anomalous"], "cause": ann["cause"],
                       "evidence": ann["evidence"], "explanation": ann["explanation"]},
                      ensure_ascii=False)


def parse_answer(text):
    """Model or teacher output -> dict, or None. Tolerates code fences and text around."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or "anomalous" not in d:
        return None
    a = d.get("anomalous")
    if isinstance(a, str):
        a = a.strip().lower() in ("true", "yes", "1")
    ev = d.get("evidence") or []
    if isinstance(ev, str):
        ev = [ev]
    return {"anomalous": bool(a), "cause": str(d.get("cause") or ""),
            "evidence": [str(e) for e in ev], "explanation": str(d.get("explanation") or "")}


def predicted_categories(ans):
    """Categories an answer talks about (set). Empty set for 'nothing wrong'."""
    if not ans or not ans["anomalous"]:
        return set()
    text = f"{ans['cause']} {ans['explanation']} {' '.join(ans['evidence'])}".lower()
    return {c for c, pat in CATEGORY_WORDS.items() if re.search(pat, text)}


def _norm(s):
    s = s.lower().replace("missing:", "").strip()
    s = re.sub(r"^\d+x\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .;")


def evidence_supported(ans, prompt):
    """Share of evidence items that can be found in the prompt (absent events must be usual ones)."""
    if not ans or not ans["evidence"]:
        return None
    p = _norm(prompt)
    ok = 0
    for e in ans["evidence"]:
        core = _norm(e)
        # long templates may be shortened by the model: check a 40-char core
        key = core[:40] if len(core) > 40 else core
        ok += bool(key) and key in p
    return ok / len(ans["evidence"])


def _rate(xs):
    """Share of blocks for which the answer claims a problem."""
    return sum(bool(x["answer"] and x["answer"]["anomalous"]) for x in xs) / len(xs) if xs else None


def score(items):
    """
    items: list of dicts with gold ('category', 'label') , 'answer' (parsed or None), 'prompt'.
    Returns a dict of metrics, computed against the gold from the raw log, not the teacher.
    """
    anom = [x for x in items if x["label"] == "Anomaly"]
    norm = [x for x in items if x["label"] == "Normal"]
    valid = [x for x in items if x["answer"] is not None]

    def cat_ok(x):
        cats = predicted_categories(x["answer"])
        return x["category"] in cats

    def invented(x):
        """Anomalous block: names some cause, but not the gold one."""
        cats = predicted_categories(x["answer"])
        return bool(cats) and x["category"] not in cats

    sup = [s for s in (evidence_supported(x["answer"], x["prompt"]) for x in valid) if s is not None]
    res = {
        "n": len(items), "n_anomaly": len(anom), "n_normal": len(norm),
        "valid_json": len(valid) / max(1, len(items)),
        # anomalous blocks
        "flags_problem": sum(bool(x["answer"] and x["answer"]["anomalous"]) for x in anom) / max(1, len(anom)),
        "correct_cause": sum(cat_ok(x) for x in anom) / max(1, len(anom)),
        "wrong_cause": sum(invented(x) for x in anom) / max(1, len(anom)),
        # normal blocks: any claimed problem is invented
        "invents_on_normal": sum(bool(x["answer"] and x["answer"]["anomalous"]) for x in norm) / max(1, len(norm)),
        "invents_on_normal_random": _rate([x for x in norm if x["category"] != "re_replication"]),
        "invents_on_normal_rereplication": _rate([x for x in norm if x["category"] == "re_replication"]),
        # evidence items found in the input
        "evidence_supported": sum(sup) / len(sup) if sup else None,
    }
    per_cat = {}
    for c in CATEGORIES:
        xs = [x for x in anom if x["category"] == c]
        if xs:
            per_cat[c] = {"n": len(xs), "correct": sum(cat_ok(x) for x in xs) / len(xs)}
    res["per_category"] = per_cat
    return res
