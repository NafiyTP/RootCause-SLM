"""
Annotates the v2 blocks with a large model (the teacher), one block per API call.

Works with any OpenAI-compatible chat API. Two presets:
  --provider groq    Llama 3.3-70B on Groq (key in GROQ_API_KEY), as in v1
  --provider gemini  Gemini through Google's OpenAI-compatible endpoint (key in GEMINI_API_KEY)

Temperature 0, JSON output, results appended to a .jsonl file as they come, so the
script can be stopped and restarted: blocks already done are skipped.

Usage:
    export GROQ_API_KEY=...
    python src/annotate_v2.py --split train
    python src/annotate_v2.py --split test
    python src/annotate_v2.py --split train --provider gemini --model gemini-2.0-flash

Then data/v2/<split>_annotated.json is written when every block is done.
"""

import argparse
import json
import os
import time

import requests

from blocks_v2 import SYSTEM_PROMPT, parse_answer
from detect import ROOT

DATA_DIR = os.path.join(ROOT, "data", "v2")
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY",
               "gemini-2.0-flash"),
}


def call(base_url, key, model, prompt, json_mode=True, retries=6):
    body = {"model": model, "temperature": 0, "max_tokens": 400,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": prompt}]}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    wait = 5
    for attempt in range(retries):
        try:
            r = requests.post(f"{base_url}/chat/completions", json=body, timeout=60,
                              headers={"Authorization": f"Bearer {key}"})
        except requests.RequestException as e:
            print(f"  network error ({e}), retry in {wait}s")
            time.sleep(wait)
            wait = min(wait * 2, 120)
            continue
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        if r.status_code in (429, 500, 502, 503):
            try:
                delay = float(r.headers.get("retry-after", wait))
            except ValueError:
                delay = wait
            if delay > 300:
                # a daily quota, not a per-minute one: stop cleanly, progress is saved
                raise SystemExit(f"rate limit reached ({r.text[:200]}). Progress is saved: "
                                 "run the same command again later, or switch provider.")
            print(f"  HTTP {r.status_code}, retry in {delay:.0f}s")
            time.sleep(delay)
            wait = min(wait * 2, 120)
            continue
        if r.status_code == 400 and json_mode:
            # some models/providers refuse response_format: ask again without it
            return call(base_url, key, model, prompt, json_mode=False, retries=retries)
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    raise RuntimeError("too many retries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--provider", choices=list(PROVIDERS), default="groq")
    ap.add_argument("--model", default=None, help="override the provider's default model")
    ap.add_argument("--sleep", type=float, default=3.0, help="seconds between calls (rate limits)")
    ap.add_argument("--limit", type=int, default=None, help="only the first n blocks (to test)")
    args = ap.parse_args()

    base_url, key_env, default_model = PROVIDERS[args.provider]
    model = args.model or default_model
    key = os.environ.get(key_env)
    if not key:
        raise SystemExit(f"set {key_env} first, e.g. export {key_env}=...")

    with open(os.path.join(DATA_DIR, f"{args.split}_blocks.json"), encoding="utf-8") as f:
        blocks = json.load(f)
    if args.limit:
        blocks = blocks[:args.limit]

    raw_path = os.path.join(DATA_DIR, f"{args.split}_raw_{args.provider}.jsonl")
    done = {}
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("answer") is not None:
                    done[r["block_id"]] = r
    print(f"{len(blocks)} blocks, {len(done)} already annotated, model {model}")

    with open(raw_path, "a", encoding="utf-8") as out:
        for i, b in enumerate(blocks):
            if b["block_id"] in done:
                continue
            text = call(base_url, key, model, b["prompt"])
            ans = parse_answer(text)
            rec = {"block_id": b["block_id"], "model": model, "raw": text, "answer": ans}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            done[b["block_id"]] = rec
            status = "ok" if ans else "INVALID JSON"
            print(f"[{i + 1}/{len(blocks)}] {b['block_id']} {b['label']:7} {status}  "
                  f"{(ans or {}).get('cause', '')[:60]}")
            time.sleep(args.sleep)

    missing = [b["block_id"] for b in blocks if done.get(b["block_id"], {}).get("answer") is None]
    if missing:
        print(f"{len(missing)} blocks without a valid answer, run the script again")
        return
    merged = [dict(b, teacher=done[b["block_id"]]["answer"], teacher_model=model) for b in blocks]
    path = os.path.join(DATA_DIR, f"{args.split}_annotated.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)
    print(f"all done, written {path}")


if __name__ == "__main__":
    main()
