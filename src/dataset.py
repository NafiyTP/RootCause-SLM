"""
dataset.py — Pipeline de données pour le fine-tuning sur logs HDFS
Modèle cible : Qwen2.5-1.5B (ou tout modèle HuggingFace compatible)

Installation :
    pip install torch transformers datasets

Usage rapide :
    from dataset import HDFSLogDataset, HDFSDataCollator
    dataset  = HDFSLogDataset("hdfs_dataset.json", tokenizer, max_length=512)
    collator = HDFSDataCollator(tokenizer)
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
#  FORMAT DU PROMPT
#  On suit le format instruction-réponse standard pour le fine-tuning causal.
#  Le modèle apprend à produire la réponse étant donné l'instruction.
#
#  Structure :
#    <|im_start|>system
#    Tu es un expert HDFS...
#    <|im_end|>
#    <|im_start|>user
#    Log : ...  Label : ...
#    <|im_end|>
#    <|im_start|>assistant
#    {"cause": ..., "raisonnement": ...}
#    <|im_end|>
#
#  Qwen2.5 utilise le format ChatML — ces tokens sont déjà dans son vocabulaire.
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Tu es un expert en systèmes distribués Hadoop/HDFS. "
    "Étant donné un log et son label (Normal ou Anomaly), "
    "tu fournis une cause technique précise et un raisonnement en 3 étapes."
)

def formater_exemple(entry: dict) -> tuple[str, str]:
    """
    Retourne (prompt, reponse) pour un exemple du dataset.

    Le prompt est la partie INSTRUCTION (ce qu'on donne au modèle).
    La réponse est la partie TARGET (ce que le modèle doit apprendre à générer).

    Séparer les deux est indispensable pour construire le masque de labels :
    on ne calcule la loss que sur la réponse, jamais sur le prompt.
    """
    prompt = (
        f"Log HDFS : {entry['log'][:300]}\n"
        f"Label : {entry['label']}"
    )
    reponse = json.dumps(
        {"cause": entry["cause"], "raisonnement": entry["raisonnement"]},
        ensure_ascii=False,
    )
    return prompt, reponse


# ─────────────────────────────────────────────────────────────────────────────
#  CLASSE DATASET
# ─────────────────────────────────────────────────────────────────────────────

class HDFSLogDataset(Dataset):
    """
    Dataset PyTorch pour le fine-tuning causal sur logs HDFS.

    Chaque exemple est tokenisé en deux temps :
      1. prompt seul  → pour connaître la longueur du prompt (n_prompt_tokens)
      2. prompt+réponse → texte complet tokenisé

    Les labels sont construits en copiant les input_ids et en masquant
    les tokens du prompt avec -100. La cross-entropy ignore les -100
    automatiquement, donc la loss n'est calculée que sur la réponse.

    Args:
        json_path   : chemin vers hdfs_dataset.json
        tokenizer   : tokenizer HuggingFace déjà chargé
        max_length  : longueur maximale (en tokens) d'un exemple
        skip_empty  : ignorer les entrées sans cause ni raisonnement
    """

    def __init__(
        self,
        json_path:  str,
        tokenizer:  Any,
        max_length: int  = 512,
        skip_empty: bool = True,
    ):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.donnees    = []

        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)

        ignores = 0
        for entry in raw:
            if skip_empty and (not entry.get("cause") or not entry.get("raisonnement")):
                ignores += 1
                continue
            self.donnees.append(entry)

        print(f"Dataset chargé : {len(self.donnees)} exemples "
              f"({ignores} ignorés car vides)")

    def __len__(self) -> int:
        return len(self.donnees)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        entry = self.donnees[idx]
        prompt, reponse = formater_exemple(entry)

        # ── Texte complet au format ChatML (natif Qwen2.5) ──────────────────
        texte_complet = self.tokenizer.apply_chat_template(
            [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": reponse},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        # ── Prompt seul (pour mesurer sa longueur en tokens) ────────────────
        texte_prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,   # ajoute <|im_start|>assistant\n
        )

        # ── Tokenisation ────────────────────────────────────────────────────
        tokenise_complet = self.tokenizer(
            texte_complet,
            max_length=self.max_length,
            truncation=True,
            padding=False,     # le padding est géré par le DataCollator
            return_tensors="pt",
        )
        tokenise_prompt = self.tokenizer(
            texte_prompt,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        input_ids      = tokenise_complet["input_ids"].squeeze(0)
        attention_mask = tokenise_complet["attention_mask"].squeeze(0)
        n_prompt       = tokenise_prompt["input_ids"].shape[1]

        # ── Construction des labels avec masquage du prompt ─────────────────
        # Les -100 disent à la cross-entropy "ignore ce token dans la loss".
        # On ne supervise que les tokens de la réponse.
        labels = input_ids.clone()
        labels[:n_prompt] = -100

        return {
            "input_ids":      input_ids,        # (seq_len,)
            "attention_mask": attention_mask,    # (seq_len,)
            "labels":         labels,            # (seq_len,)  — prompt masqué
        }


# ─────────────────────────────────────────────────────────────────────────────
#  DATA COLLATOR
#  Regroupe les exemples d'un batch et les aligne par padding à droite.
#
#  Pourquoi padding à droite pour le training ?
#  Le modèle causal calcule des positions relatives : le token i ne voit
#  que les tokens 0..i-1. Si on padde à gauche, les positions de vrais
#  tokens changent entre exemples, ce qui perturbe les embeddings positionnels.
#  Le padding à droite garantit que les positions restent cohérentes.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HDFSDataCollator:
    """
    Collator qui padde dynamiquement les exemples d'un batch.

    - input_ids      : paddé avec pad_token_id
    - attention_mask : paddé avec 0 (les tokens de padding sont ignorés)
    - labels         : paddé avec -100 (la loss ignore le padding)

    Le padding est dynamique : on padde jusqu'au plus long exemple du batch,
    pas jusqu'à max_length. Ça économise de la mémoire GPU.
    """

    tokenizer: Any

    def __call__(self, exemples: list[dict]) -> dict[str, torch.Tensor]:
        input_ids_list      = [e["input_ids"]      for e in exemples]
        attention_mask_list = [e["attention_mask"]  for e in exemples]
        labels_list         = [e["labels"]          for e in exemples]

        # Longueur du plus long exemple dans ce batch
        max_len = max(ids.shape[0] for ids in input_ids_list)

        input_ids_padded      = []
        attention_mask_padded = []
        labels_padded         = []

        for ids, mask, lbl in zip(input_ids_list, attention_mask_list, labels_list):
            n_pad = max_len - ids.shape[0]

            # Padding à droite
            ids_p  = torch.cat([ids,  torch.full((n_pad,), self.tokenizer.pad_token_id)])
            mask_p = torch.cat([mask, torch.zeros(n_pad, dtype=torch.long)])
            lbl_p  = torch.cat([lbl,  torch.full((n_pad,), -100)])

            input_ids_padded.append(ids_p)
            attention_mask_padded.append(mask_p)
            labels_padded.append(lbl_p)

        return {
            "input_ids":      torch.stack(input_ids_padded),       # (B, max_len)
            "attention_mask": torch.stack(attention_mask_padded),   # (B, max_len)
            "labels":         torch.stack(labels_padded),           # (B, max_len)
        }


# ─────────────────────────────────────────────────────────────────────────────
#  VÉRIFICATION RAPIDE
#  Lance ce fichier directement pour vérifier que tout fonctionne :
#      python dataset.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    JSON_PATH  = "hdfs_dataset.json"

    print(f"Chargement du tokenizer : {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Qwen2.5 a déjà un pad_token mais on s'assure qu'il est défini
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset  = HDFSLogDataset(JSON_PATH, tokenizer, max_length=512)
    collator = HDFSDataCollator(tokenizer)

    # ── DataLoader ───────────────────────────────────────────────────────────
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collator)

    # ── Vérification sur le premier batch ────────────────────────────────────
    batch = next(iter(loader))

    print("\n── Vérification du premier batch ──────────────────────────────")
    print(f"  input_ids      : {batch['input_ids'].shape}")       # (4, seq_len)
    print(f"  attention_mask : {batch['attention_mask'].shape}")
    print(f"  labels         : {batch['labels'].shape}")

    # Vérifier que le masquage du prompt est correct
    premier      = batch["labels"][0]
    n_masques    = (premier == -100).sum().item()
    n_supervises = (premier != -100).sum().item()
    print(f"\n  Exemple 0 — tokens masqués (prompt) : {n_masques}")
    print(f"  Exemple 0 — tokens supervisés (réponse) : {n_supervises}")

    # Décoder la partie réponse pour vérification visuelle
    ids_reponse = batch["input_ids"][0][premier != -100]
    print(f"\n  Réponse décodée :\n  {tokenizer.decode(ids_reponse)}")

    print("\n✅ Pipeline OK — prêt pour le fine-tuning")