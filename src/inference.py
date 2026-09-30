"""
Explain an HDFS log line with the fine-tuned model.

The model was trained with the label as input (log + Normal/Anomaly -> cause and
reasoning), so it explains a label, it does not predict one. The label has to come
from somewhere else, e.g. a separate detector.

Usage:
    python src/inference.py                          # built-in examples
    python src/inference.py --log "081109 ..." --label Anomaly
    python src/inference.py --file logs.txt --label Anomaly   # one log per line
"""

import argparse
import json
import os
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import SYSTEM_PROMPT, format_example

BASE_MODEL     = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modele_hdfs")
MAX_NEW_TOKENS = 300
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

EXAMPLES = [
    ("081109 203615 148 WARN dfs.DataNode$DataXceiver: Got exception while serving "
     "blk_-6952295868487656571 to /10.251.73.220", "Anomaly"),
    ("081109 204005 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap "
     "updated: 10.251.73.220:50010 is added to blk_7128370237687728475", "Normal"),
    ("081109 204525 512 INFO dfs.DataNode$PacketResponder: PacketResponder 2 for block "
     "blk_572492839287299681 terminating", "Normal"),
]


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map={"": DEVICE},
    )
    model = PeftModel.from_pretrained(model, LORA_DIR)
    model.eval()
    return model, tokenizer


def explain(log: str, label: str, model, tokenizer) -> dict:
    # same prompt as in training (src/dataset.py)
    prompt, _ = format_example({"log": log, "label": label, "cause": "", "raisonnement": ""})
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    cleaned = re.sub(r"^```(?:json)?", "", raw).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        result = {"cause": None, "raisonnement": None}
    result["raw"] = raw
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str)
    parser.add_argument("--file", type=str, help="text file, one log line per line")
    parser.add_argument("--label", choices=["Normal", "Anomaly"], default="Anomaly")
    args = parser.parse_args()

    if args.log:
        items = [(args.log, args.label)]
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            items = [(line.strip(), args.label) for line in f if line.strip()]
    else:
        items = EXAMPLES

    model, tokenizer = load_model()
    for log, label in items:
        r = explain(log, label, model, tokenizer)
        print(f"\n[{label}] {log[:100]}")
        if r["cause"] is None:
            print(f"  could not parse output: {r['raw'][:200]}")
        else:
            print(f"  cause:     {r['cause']}")
            print(f"  reasoning: {r['raisonnement']}")


if __name__ == "__main__":
    main()
