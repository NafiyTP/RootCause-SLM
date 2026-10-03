"""
How fast and how expensive is the fine-tuned 1.5B model, compared to calling the
70B teacher through an API?

Measured on the test set (GPU):
  - latency per line with batch size 1 (one request at a time)
  - throughput with batching (lines/s and generated tokens/s)
  - peak GPU memory

Cost, with the prices below:
  - fine-tuned model: GPU price per hour / lines per hour
  - teacher: tokens of each request x Groq price per token. Tokens are counted with
    the Qwen tokenizer on the same prompt and the reference answer, so it is an
    estimate (Llama 3 tokenizes a bit differently, and the real annotation prompt was
    a bit longer).

Usage:
    python src/benchmark.py              # 64 lines, batch sizes 1 and 16
    python src/benchmark.py --n 128 --batch-sizes 1 8 32
"""

import argparse
import json
import os
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import LORA_DIR, ROOT, TEST_JSON, build_messages, load, target_json

# Prices in USD, to update if they change
GPU_PRICE_PER_HOUR = 0.35     # NVIDIA T4, on-demand on Google Cloud
TEACHER_INPUT_PER_M = 0.59    # Groq, llama-3.3-70b-versatile, per 1M input tokens
TEACHER_OUTPUT_PER_M = 0.79   # per 1M output tokens

OUT_DIR = os.path.join(ROOT, "results", "benchmark")


def run(model, tok, entries, batch_size, max_new, stop_ids):
    """Generate for all entries, returns (seconds, number of generated tokens)."""
    n_tokens = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for b in range(0, len(entries), batch_size):
        batch = entries[b:b + batch_size]
        prompts = [tok.apply_chat_template(build_messages(e, "finetuned", None),
                                           tokenize=False, add_generation_prompt=True)
                   for e in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=stop_ids)
        gen = out[:, enc["input_ids"].shape[1]:]
        n_tokens += int((gen != tok.pad_token_id).sum())
    torch.cuda.synchronize()
    return time.perf_counter() - t0, n_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "this benchmark needs a GPU"

    test = load(TEST_JSON)[:args.n]

    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16).to("cuda")
    model = PeftModel.from_pretrained(model, LORA_DIR)
    model = model.merge_and_unload()  # merge LoRA into the weights, no extra cost at inference
    model.eval()
    stop_ids = [i for i in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
                if isinstance(i, int) and i >= 0]

    run(model, tok, test[:4], 4, 32, stop_ids)  # warm-up
    torch.cuda.reset_peak_memory_stats()

    gpu = torch.cuda.get_device_name(0)
    rows = []
    for bs in args.batch_sizes:
        secs, n_tok = run(model, tok, test, bs, args.max_new_tokens, stop_ids)
        lines_per_s = len(test) / secs
        cost_1k = GPU_PRICE_PER_HOUR / (lines_per_s * 3600) * 1000
        rows.append({"batch_size": bs, "s_per_line": round(secs / len(test), 3),
                     "lines_per_s": round(lines_per_s, 2), "tokens_per_s": round(n_tok / secs, 1),
                     "usd_per_1k_lines": round(cost_1k, 4)})
        print(rows[-1])
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    # teacher cost, estimated from token counts
    n_in = n_out = 0
    for e in test:
        n_in += len(tok.apply_chat_template(build_messages(e, "finetuned", None), tokenize=True))
        n_out += len(tok(target_json(e))["input_ids"])
    teacher_1k = (n_in * TEACHER_INPUT_PER_M + n_out * TEACHER_OUTPUT_PER_M) / 1e6 / len(test) * 1000

    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {"gpu": gpu, "n_lines": len(test), "peak_gpu_memory_gb": round(peak_gb, 2),
               "local": rows, "teacher_usd_per_1k_lines": round(teacher_1k, 4),
               "avg_tokens_in": round(n_in / len(test)), "avg_tokens_out": round(n_out / len(test)),
               "prices": {"gpu_per_hour": GPU_PRICE_PER_HOUR, "teacher_in_per_m": TEACHER_INPUT_PER_M,
                          "teacher_out_per_m": TEACHER_OUTPUT_PER_M}}
    with open(os.path.join(OUT_DIR, "benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md = [f"# Inference benchmark ({gpu}, {len(test)} test lines, fp16, LoRA merged)", "",
          "| Setup | s / line | lines / s | tokens / s | USD per 1k lines |",
          "|---|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| Qwen2.5-1.5B + LoRA, batch {r['batch_size']} | {r['s_per_line']:.3f} | "
                  f"{r['lines_per_s']:.2f} | {r['tokens_per_s']:.0f} | {r['usd_per_1k_lines']:.4f} |")
    md.append(f"| Llama 3.3-70B on Groq (estimate) | | | | {teacher_1k:.4f} |")
    md += ["", f"Peak GPU memory: {peak_gb:.2f} GB. Average request: "
           f"{n_in // len(test)} tokens in, {n_out // len(test)} tokens out.",
           f"Prices: GPU {GPU_PRICE_PER_HOUR} $/h, Groq {TEACHER_INPUT_PER_M} / "
           f"{TEACHER_OUTPUT_PER_M} $ per 1M tokens in / out."]
    with open(os.path.join(OUT_DIR, "benchmark.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
