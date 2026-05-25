"""
train.py — Fine-tuning LoRA de Qwen2.5-1.5B sur logs HDFS
Gère le déséquilibre Normal/Anomaly via WeightedRandomSampler.

Installation sur Colab (cellule séparée) :
    !pip install transformers peft accelerate -q

Usage :
    python train.py
"""

import json
import math
import os
import torch
import numpy as np

from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from dataset import HDFSLogDataset, HDFSDataCollator

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME    = "Qwen/Qwen2.5-1.5B-Instruct"
JSON_PATH     = "hdfs_dataset.json"
OUTPUT_DIR    = "/content/modele_hdfs"

# Entraînement
MAX_LENGTH    = 512
BATCH_SIZE    = 4
GRAD_ACCUM    = 4          # batch effectif = BATCH_SIZE × GRAD_ACCUM = 16
NUM_EPOCHS    = 5
LR            = 2e-4
WARMUP_RATIO  = 0.05       # 5% des steps pour le warmup
VAL_RATIO     = 0.15       # 15% du dataset pour la validation
SEED          = 42

# LoRA
LORA_R        = 16         # rang des matrices LoRA
LORA_ALPHA    = 32         # facteur d'échelle = alpha/r = 2
LORA_DROPOUT  = 0.05
LORA_TARGETS  = ["q_proj", "k_proj", "v_proj", "o_proj"]  # couches d'attention

torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

# ─────────────────────────────────────────────────────────────────────────────
#  1. TOKENIZER
# ─────────────────────────────────────────────────────────────────────────────
print("📥 Chargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ─────────────────────────────────────────────────────────────────────────────
#  2. DATASET + SPLIT TRAIN / VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
print("📥 Chargement du dataset...")
dataset_complet = HDFSLogDataset(JSON_PATH, tokenizer, max_length=MAX_LENGTH)

n_val   = int(len(dataset_complet) * VAL_RATIO)
n_train = len(dataset_complet) - n_val

dataset_train, dataset_val = random_split(
    dataset_complet,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED),
)
print(f"   Train : {n_train} exemples | Val : {n_val} exemples\n")

# ─────────────────────────────────────────────────────────────────────────────
#  3. WEIGHTED RANDOM SAMPLER
#
#  Problème : 96.5% Normal / 3.5% Anomaly
#  Solution : on donne un poids inversement proportionnel à la fréquence
#             de chaque classe, pour que le sampler tire autant de Normal
#             que d'Anomaly dans chaque batch.
#
#  Poids par classe :
#    w_Normal  = 1 / n_Normal
#    w_Anomaly = 1 / n_Anomaly  (≈ 28x plus grand)
#
#  Chaque exemple reçoit le poids de sa classe → le sampler tire
#  proportionnellement à ces poids avec remise.
# ─────────────────────────────────────────────────────────────────────────────
print("⚖️  Construction du WeightedRandomSampler...")

# Récupérer les labels des exemples d'entraînement
labels_train = [
    dataset_complet.donnees[idx]["label"]
    for idx in dataset_train.indices
]

n_normal  = labels_train.count("Normal")
n_anomaly = labels_train.count("Anomaly")
print(f"   Normal : {n_normal} | Anomaly : {n_anomaly}")
print(f"   Ratio brut : {n_normal/n_anomaly:.1f}:1 → corrigé à ~1:1 par le sampler")

poids_par_classe = {
    "Normal":  1.0 / n_normal,
    "Anomaly": 1.0 / n_anomaly,
}
poids_exemples = torch.tensor(
    [poids_par_classe[l] for l in labels_train],
    dtype=torch.float,
)

sampler = WeightedRandomSampler(
    weights=poids_exemples,
    num_samples=len(poids_exemples),
    replacement=True,   # tirage avec remise — indispensable pour rééquilibrer
)
print("   ✓ Sampler prêt\n")

# ─────────────────────────────────────────────────────────────────────────────
#  4. DATALOADERS
# ─────────────────────────────────────────────────────────────────────────────
collator = HDFSDataCollator(tokenizer)

loader_train = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE,
    sampler=sampler,        # WeightedRandomSampler remplace shuffle=True
    collate_fn=collator,
    pin_memory=(DEVICE == "cuda"),
)
loader_val = DataLoader(
    dataset_val,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collator,
    pin_memory=(DEVICE == "cuda"),
)

# ─────────────────────────────────────────────────────────────────────────────
#  5. MODÈLE + LORA
#
#  LoRA (Low-Rank Adaptation) : au lieu de mettre à jour W (d×d),
#  on apprend deux petites matrices A (d×r) et B (r×d) telles que
#  ΔW = B·A avec r << d. Ici r=16, d≈2048 → on entraîne 16/2048 ≈ 0.8%
#  des paramètres originaux.
#
#  alpha/r = 32/16 = 2 : facteur d'échelle appliqué à ΔW.
#  Un ratio de 2 est standard — il équilibre la contribution de LoRA
#  avec les poids pré-entraînés.
# ─────────────────────────────────────────────────────────────────────────────
print("🏗️  Chargement du modèle...")
# APRÈS
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map={"": DEVICE},   # force tout sur cuda:0 explicitement
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGETS,
    bias="none",
)
model = get_peft_model(model, lora_config)

# Résumé des paramètres entraînables
params_total     = sum(p.numel() for p in model.parameters())
params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Paramètres totaux     : {params_total/1e6:.1f}M")
print(f"   Paramètres entraînés  : {params_trainable/1e6:.2f}M "
      f"({100*params_trainable/params_total:.2f}%)\n")

# ─────────────────────────────────────────────────────────────────────────────
#  6. OPTIMISEUR + SCHEDULER
#
#  AdamW avec cosine schedule + warmup :
#  - Warmup : le LR monte linéairement pendant 5% des steps (le modèle
#    est fragile au début, on évite des mises à jour trop agressives)
#  - Cosine decay : le LR descend en cosinus jusqu'à 0 — plus doux
#    qu'une descente linéaire, évite les oscillations en fin d'entraînement
# ─────────────────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR,
    weight_decay=0.01,
)

total_steps  = math.ceil(n_train / (BATCH_SIZE * GRAD_ACCUM)) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

print(f"🔧 Entraînement")
print(f"   Steps totaux  : {total_steps}")
print(f"   Warmup steps  : {warmup_steps}")
print(f"   Batch effectif : {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM}\n")

# ─────────────────────────────────────────────────────────────────────────────
#  7. BOUCLE D'ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────
def evaluer(model, loader, device) -> float:
    """Calcule la loss moyenne sur le set de validation."""
    model.eval()
    loss_totale = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch)
            loss_totale += out.loss.item()
    return loss_totale / len(loader)


os.makedirs(OUTPUT_DIR, exist_ok=True)
meilleure_val_loss = float("inf")
historique         = []

print("=" * 72)
for epoch in range(1, NUM_EPOCHS + 1):
    # ── Training ─────────────────────────────────────────────────────────────
    model.train()
    loss_train   = 0.0
    step_global  = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader_train):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out   = model(**batch)
        loss  = out.loss / GRAD_ACCUM   # normalisation pour l'accumulation
        loss.backward()
        loss_train += out.loss.item()

        # Mise à jour des poids tous les GRAD_ACCUM steps
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step_global += 1

    loss_train_moy = loss_train / len(loader_train)

    # ── Validation ───────────────────────────────────────────────────────────
    loss_val = evaluer(model, loader_val, DEVICE)

    # Perplexité = e^loss — plus lisible que la loss brute
    ppl_train = math.exp(loss_train_moy)
    ppl_val   = math.exp(loss_val)

    print(
        f"  Epoch {epoch}/{NUM_EPOCHS} | "
        f"Loss train={loss_train_moy:.4f} (ppl={ppl_train:.1f}) | "
        f"Loss val={loss_val:.4f} (ppl={ppl_val:.1f}) | "
        f"LR={scheduler.get_last_lr()[0]:.2e}"
    )

    historique.append({
        "epoch": epoch,
        "loss_train": loss_train_moy,
        "loss_val":   loss_val,
        "ppl_train":  ppl_train,
        "ppl_val":    ppl_val,
    })

    # ── Sauvegarde du meilleur modèle ────────────────────────────────────────
    if loss_val < meilleure_val_loss:
        meilleure_val_loss = loss_val
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"   💾 Meilleur modèle sauvegardé (val_loss={loss_val:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
#  8. RÉSUMÉ
# ─────────────────────────────────────────────────────────────────────────────
with open(os.path.join(OUTPUT_DIR, "historique.json"), "w") as f:
    json.dump(historique, f, indent=2)

print(f"\n{'='*72}")
print(f"✅ Entraînement terminé")
print(f"   Meilleure val_loss : {meilleure_val_loss:.4f} "
      f"(ppl={math.exp(meilleure_val_loss):.1f})")
print(f"   Modèle sauvegardé  : {OUTPUT_DIR}/")
