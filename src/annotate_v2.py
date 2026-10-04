"""
Annotates the v2 blocks with a large model (the teacher), one block per API call.

Works with any OpenAI-compatible chat API. Two presets:
  --provider groq    GPT-OSS 120B on Groq by default (key in GROQ_API_KEY). v1 used
                     llama-3.3-70b-versatile, which not every account can access
                     (--model llama-3.3-70b-versatile to use it)
  --provider gemini  Gemini through Google's OpenAI-compatible endpoint (key in GEMINI_API_KEY)

Temperature 0, JSON output, results appended to a .jsonl file as they come, so the
script can be stopped and restarted: blocks already done are skipped.

Usage:
    export GROQ_API_KEY=key1            # or key1,key2,key3: used in turn when one hits a limit
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
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY",
               "gemini-2.0-flash"),
}


def list_models(base_url, key):
    try:
        r = requests.get(f"{base_url}/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
        return sorted(m["id"] for m in r.json().get("data", []))
    except (requests.RequestException, ValueError, KeyError):
        return ["(could not list models)"]


class KeyBlocked(Exception):
    """This key cannot be used now: rate limit (with the delay in seconds) or no credit left."""

    def __init__(self, delay, message):
        super().__init__(message)
        self.delay = delay


class KeyPool:
    """
    Several keys of the same account, used in turn. A key that hits a limit is put aside
    until its limit resets, and the next free key is used. When every key is blocked,
    wait if the shortest block is short, otherwise stop (progress is saved).
    """

    def __init__(self, keys):
        self.keys = keys
        self.blocked_until = [0.0] * len(keys)
        self.i = 0

    def current(self):
        now = time.time()
        for k in range(len(self.keys)):
            j = (self.i + k) % len(self.keys)
            if self.blocked_until[j] <= now:
                if j != self.i:
                    print(f"  switching to key {j + 1}/{len(self.keys)}")
                self.i = j
                return self.keys[j]
        wait = min(self.blocked_until) - now
        if wait > 300:
            raise SystemExit("every key is blocked (limits or credit). Progress is saved: "
                             "run the same command again later, or add a key.")
        print(f"  all keys rate-limited, waiting {wait:.0f}s")
        time.sleep(wait)
        return self.current()

    def block(self, delay):
        self.blocked_until[self.i] = time.time() + delay


def call(base_url, key, model, prompt, json_mode=True, retries=6):
    body = {"model": model, "temperature": 0, "max_tokens": 400,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": prompt}]}
    if "gpt-oss" in model:
        # reasoning model: keep the hidden reasoning short and leave room for the answer
        body["reasoning_effort"] = "low"
        body["max_tokens"] = 2000
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
            if r.status_code == 429:
                # let the caller try another key (or wait / stop if there is none)
                raise KeyBlocked(delay, r.text[:200])
            print(f"  HTTP {r.status_code}, retry in {delay:.0f}s")
            time.sleep(delay)
            wait = min(wait * 2, 120)
            continue
        if r.status_code in (401, 402, 403):
            # invalid key or no credit left on it: put it aside for a day
            raise KeyBlocked(86400, f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code == 404 and "model" in r.text:
            raise SystemExit(f"model {model} not available for this key ({r.text[:200]}).\n"
                             f"Available models: {', '.join(list_models(base_url, key))}\n"
                             "Pick one with --model.")
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
    ap.add_argument("--list-models", action="store_true", help="print the models this key can use")
    args = ap.parse_args()

    base_url, key_env, default_model = PROVIDERS[args.provider]
    model = args.model or default_model
    # one key, or several separated by commas (same account, e.g. several prepaid keys)
    keys = [k.strip() for k in os.environ.get(key_env, "").split(",") if k.strip()]
    if not keys:
        raise SystemExit(f"set {key_env} first, e.g. export {key_env}=key1,key2")
    if args.list_models:
        print("\n".join(list_models(base_url, keys[0])))
        return
    pool = KeyPool(keys)

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
    print(f"{len(blocks)} blocks, {len(done)} already annotated, model {model}, {len(keys)} key(s)")

    with open(raw_path, "a", encoding="utf-8") as out:
        for i, b in enumerate(blocks):
            if b["block_id"] in done:
                continue
            while True:
                try:
                    text = call(base_url, pool.current(), model, b["prompt"])
                    break
                except KeyBlocked as e:
                    print(f"  key {pool.i + 1} blocked for {e.delay:.0f}s ({e})")
                    pool.block(e.delay)
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
