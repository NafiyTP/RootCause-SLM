"""
LoRA fine-tuning of Qwen2.5-1.5B-Instruct on the v2 data: block summary in, teacher's
JSON answer out. Same recipe as src/train.py (LoRA r=16 on q/k/v/o, loss on the answer
tokens only, AdamW 2e-4 with warm-up and cosine decay), with longer inputs.

The training set is already stratified by category (build_dataset_v2.py), so no
weighted sampler here.

Run on a GPU (Colab T4):
    python src/train_v2.py
Adapter saved in modele_hdfs_v2/.
"""

import json
import math
import os

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from blocks_v2 import SYSTEM_PROMPT, target_json
from dataset import HDFSDataCollator

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
JSON_PATH  = os.path.join(SRC_DIR, "..", "data", "v2", "train_annotated.json")
OUTPUT_DIR = os.path.join(SRC_DIR, "..", "modele_hdfs_v2")

MAX_LENGTH   = 1536
BATCH_SIZE   = 2
GRAD_ACCUM   = 8      # effective batch size 16
NUM_EPOCHS   = 5
LR           = 2e-4
WARMUP_RATIO = 0.05
VAL_RATIO    = 0.15
SEED         = 42

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


class BlockDataset(Dataset):
    """ChatML system + block summary + teacher JSON; labels are -100 on the prompt."""

    def __init__(self, path, tokenizer, max_length):
        with open(path, encoding="utf-8") as f:
            self.data = [d for d in json.load(f) if d.get("teacher")]
        self.tok = tokenizer
        self.max_length = max_length
        print(f"Loaded {len(self.data)} annotated blocks")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": d["prompt"]}]
        prompt_text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full_text = self.tok.apply_chat_template(
            msgs + [{"role": "assistant", "content": target_json(d["teacher"])}],
            tokenize=False, add_generation_prompt=False)
        full = self.tok(full_text, max_length=self.max_length, truncation=True, return_tensors="pt")
        n_prompt = self.tok(prompt_text, max_length=self.max_length, truncation=True,
                            return_tensors="pt")["input_ids"].shape[1]
        input_ids = full["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[:n_prompt] = -100
        return {"input_ids": input_ids, "attention_mask": full["attention_mask"].squeeze(0),
                "labels": labels}


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    full = BlockDataset(JSON_PATH, tok, MAX_LENGTH)
    n_val = int(len(full) * VAL_RATIO)
    train_set, val_set = random_split(full, [len(full) - n_val, n_val],
                                      generator=torch.Generator().manual_seed(SEED))
    collator = HDFSDataCollator(tok)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)
    print(f"train: {len(train_set)}, val: {len(val_set)}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map={"": device})
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS, bias="none"))
    model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    total = math.ceil(len(train_set) / (BATCH_SIZE * GRAD_ACCUM)) * NUM_EPOCHS
    sched = get_cosine_schedule_with_warmup(opt, int(total * WARMUP_RATIO), total)

    def val_loss():
        model.eval()
        s = 0.0
        with torch.no_grad():
            for b in val_loader:
                s += model(**{k: v.to(device) for k, v in b.items()}).loss.item()
        return s / len(val_loader)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best, history = float("inf"), []
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        tl = 0.0
        opt.zero_grad()
        for step, b in enumerate(train_loader):
            out = model(**{k: v.to(device) for k, v in b.items()})
            (out.loss / GRAD_ACCUM).backward()
            tl += out.loss.item()
            if (step + 1) % GRAD_ACCUM == 0 or step + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
        tl /= len(train_loader)
        vl = val_loss()
        print(f"epoch {epoch}/{NUM_EPOCHS} | train {tl:.4f} | val {vl:.4f} (ppl {math.exp(vl):.2f})")
        history.append({"epoch": epoch, "loss_train": tl, "loss_val": vl, "ppl_val": math.exp(vl)})
        if vl < best:
            best = vl
            model.save_pretrained(OUTPUT_DIR)
            tok.save_pretrained(OUTPUT_DIR)
            print("  saved best adapter")
    with open(os.path.join(OUTPUT_DIR, "historique.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"done, best val loss {best:.4f}, adapter in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
