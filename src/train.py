"""
LoRA fine-tuning of Qwen2.5-1.5B-Instruct on the annotated HDFS logs.

Only 3.5% of the training lines are anomalies, so I oversample them with a
WeightedRandomSampler to get roughly balanced batches.

I trained on a Colab T4:
    !pip install -r requirements.txt
    !python src/train.py
"""

import json
import math
import os

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import HDFSDataCollator, HDFSLogDataset

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
JSON_PATH  = os.path.join(SRC_DIR, "..", "data", "hdfs_dataset.json")
OUTPUT_DIR = os.path.join(SRC_DIR, "..", "modele_hdfs")

MAX_LENGTH   = 512
BATCH_SIZE   = 4
GRAD_ACCUM   = 4      # effective batch size 16
NUM_EPOCHS   = 5
LR           = 2e-4
WARMUP_RATIO = 0.05
VAL_RATIO    = 0.15
SEED         = 42

# LoRA on the attention projections only. alpha / r = 2 scales the update.
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")
if DEVICE == "cuda":
    print(f"gpu: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ---------------------------------------------------------------- data

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

full_dataset = HDFSLogDataset(JSON_PATH, tokenizer, max_length=MAX_LENGTH)
n_val = int(len(full_dataset) * VAL_RATIO)
n_train = len(full_dataset) - n_val
train_set, val_set = random_split(
    full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
)
print(f"train: {n_train}, val: {n_val}")

# Each example gets weight 1 / (size of its class), sampled with replacement,
# so Normal and Anomaly are drawn about equally often.
train_labels = [full_dataset.data[i]["label"] for i in train_set.indices]
n_normal = train_labels.count("Normal")
n_anomaly = train_labels.count("Anomaly")
print(f"Normal: {n_normal}, Anomaly: {n_anomaly} (ratio {n_normal / n_anomaly:.1f}:1)")

class_weight = {"Normal": 1.0 / n_normal, "Anomaly": 1.0 / n_anomaly}
sample_weights = torch.tensor([class_weight[l] for l in train_labels], dtype=torch.float)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

collator = HDFSDataCollator(tokenizer)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler,
                          collate_fn=collator, pin_memory=(DEVICE == "cuda"))
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collator, pin_memory=(DEVICE == "cuda"))

# ---------------------------------------------------------------- model

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map={"": DEVICE},
)
model = get_peft_model(model, LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGETS,
    bias="none",
))

n_total = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable params: {n_trainable / 1e6:.2f}M / {n_total / 1e6:.1f}M "
      f"({100 * n_trainable / n_total:.2f}%)")

# AdamW, linear warmup on the first 5% of steps, then cosine decay to 0
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=LR, weight_decay=0.01)
total_steps = math.ceil(n_train / (BATCH_SIZE * GRAD_ACCUM)) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                            num_training_steps=total_steps)
print(f"steps: {total_steps} ({warmup_steps} warmup)")

# ---------------------------------------------------------------- training


def eval_loss(model, loader, device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            total += model(**batch).loss.item()
    return total / len(loader)


os.makedirs(OUTPUT_DIR, exist_ok=True)
best_val_loss = float("inf")
history = []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch)
        (out.loss / GRAD_ACCUM).backward()
        train_loss += out.loss.item()

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    train_loss /= len(train_loader)
    val_loss = eval_loss(model, val_loader, DEVICE)
    ppl_train, ppl_val = math.exp(train_loss), math.exp(val_loss)

    print(f"epoch {epoch}/{NUM_EPOCHS} | train {train_loss:.4f} (ppl {ppl_train:.2f}) | "
          f"val {val_loss:.4f} (ppl {ppl_val:.2f}) | lr {scheduler.get_last_lr()[0]:.2e}")

    # key names kept as in the saved modele_hdfs/historique.json
    history.append({"epoch": epoch, "loss_train": train_loss, "loss_val": val_loss,
                    "ppl_train": ppl_train, "ppl_val": ppl_val})

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"  saved best adapter (val loss {val_loss:.4f})")

with open(os.path.join(OUTPUT_DIR, "historique.json"), "w") as f:
    json.dump(history, f, indent=2)

print(f"done, best val loss {best_val_loss:.4f} (ppl {math.exp(best_val_loss):.2f}), "
      f"adapter in {OUTPUT_DIR}")
