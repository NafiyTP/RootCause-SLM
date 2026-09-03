"""
evaluate.py — Évaluation du modèle fine-tuné sur hdfs_test_dataset.json

Le modèle est un annotateur : il prend log + label → génère {cause, raisonnement}.
On compare le texte généré avec la référence Llama 70B via :
  - ROUGE-L      : chevauchement de séquences (mots exacts)
  - BERTScore    : similarité sémantique (embeddings)
  - JSON validity : taux de JSON valides générés

On compare aussi avec deux baselines :
  - Baseline 1 : répéter la cause la plus fréquente du training set
  - Baseline 2 : Qwen2.5-1.5B zero-shot (sans fine-tuning)

Le script sauvegarde les résultats de chaque modèle au fur et à mesure dans
results/resultats_evaluation.json. Si le script est relancé après un crash,
il recharge ce fichier et saute les modèles déjà évalués — aucun calcul
n'est perdu.

Usage :
    python evaluate.py
    python evaluate.py --n 50   # tester sur 50 exemples seulement
"""

import json
import re
import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# ────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
import os

BASE_MODEL      = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
LORA_DIR        = os.path.join(BASE_DIR, "..", "modele_hdfs")
TEST_JSON       = os.path.join(BASE_DIR, "..", "data", "hdfs_test_dataset.json")
TRAIN_JSON      = os.path.join(BASE_DIR, "..", "data", "hdfs_dataset.json")  # pour la baseline fréquence
RESULTS_DIR     = os.path.join(BASE_DIR, "..", "results")
MAX_NEW_TOKENS  = 250
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = (
    "Tu es un expert en systèmes distribués Hadoop/HDFS. "
    "Étant donné un log et son label, tu fournis une cause technique "
    "précise et un raisonnement en 3 étapes."
)

# ─────────────────────────────────────────────────────────────────────────────
#  1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────
def charger_test(json_path: str, n: int = None) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    data = [d for d in data if d.get("cause") and d.get("raisonnement")]
    if n:
        data = data[:n]
    print(f"   ✓ {len(data)} exemples de test chargés")
    return data


def cause_la_plus_frequente(train_json: str) -> str:
    """Baseline naïve : retourner toujours la cause la plus fréquente du training."""
    with open(train_json, encoding="utf-8") as f:
        data = json.load(f)
    from collections import Counter
    causes = [d["cause"] for d in data if d.get("cause")]
    return Counter(causes).most_common(1)[0][0]


# ─────────────────────────────────────────────────────────────────────────────
#  2. CHARGEMENT DES MODÈLES
# ─────────────────────────────────────────────────────────────────────────────
def charger_modele(lora_dir: str | None = None):
    label = "fine-tuné" if lora_dir else "zero-shot"
    print(f"📥 Chargement de Qwen2.5-1.5B ({label})...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map={"": DEVICE},
    )
    if lora_dir:
        model = PeftModel.from_pretrained(model, lora_dir)

    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
#  3. GÉNÉRATION
# ─────────────────────────────────────────────────────────────────────────────
def generer(log: str, label: str, model, tokenizer) -> dict:
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Log HDFS : {log[:300]}\n"
                    f"Label : {label}\n\n"
                    f"Réponds en JSON pur sans backticks :\n"
                    f'{{"cause": "une phrase courte", '
                    f'"raisonnement": "Étape 1 : ... Étape 2 : ... Étape 3 : ..."}}'
                ),
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    texte = tokenizer.decode(generated, skip_special_tokens=True).strip()
    texte = re.sub(r"^```(?:json)?", "", texte).strip().rstrip("```").strip()

    try:
        return {"ok": True, **json.loads(texte)}
    except json.JSONDecodeError:
        return {"ok": False, "cause": texte[:200], "raisonnement": ""}


# ─────────────────────────────────────────────────────────────────────────────
#  4. MÉTRIQUES
# ─────────────────────────────────────────────────────────────────────────────
def calculer_rouge(predictions: list[str], references: list[str]) -> float:
    """ROUGE-L : plus long sous-séquence commune, normalisée."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return round(float(np.mean(scores)), 4)


def calculer_bertscore(predictions: list[str], references: list[str]) -> float:
    """BERTScore F1 : similarité sémantique entre les embeddings."""
    _, _, F1 = bert_score(
        predictions, references,
        lang="fr",
        device=DEVICE,
        verbose=False,
    )
    return round(float(F1.mean()), 4)


def evaluer(
    nom: str,
    causes_pred: list[str],
    raison_pred: list[str],
    causes_ref:  list[str],
    raison_ref:  list[str],
    n_valides:   int,
    n_total:     int,
) -> dict:
    print(f"\n   Calcul ROUGE-L...")
    rouge_cause  = calculer_rouge(causes_pred, causes_ref)
    rouge_raison = calculer_rouge(raison_pred, raison_ref)

    print(f"   Calcul BERTScore...")
    bert_cause  = calculer_bertscore(causes_pred, causes_ref)
    bert_raison = calculer_bertscore(raison_pred, raison_ref)

    return {
        "nom":          nom,
        "json_valid":   round(n_valides / n_total, 4),
        "rouge_cause":  rouge_cause,
        "rouge_raison": rouge_raison,
        "bert_cause":   bert_cause,
        "bert_raison":  bert_raison,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  5. PIPELINE D'ÉVALUATION D'UN MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
def evaluer_modele(nom: str, model, tokenizer, test_data: list[dict]) -> dict:
    print(f"\n🔄 Génération : {nom} ({len(test_data)} exemples)...")

    causes_pred, raison_pred = [], []
    causes_ref,  raison_ref  = [], []
    n_valides = 0

    for i, entry in enumerate(test_data):
        result = generer(entry["log"], entry["label"], model, tokenizer)

        if result["ok"]:
            n_valides += 1

        causes_pred.append(result.get("cause", "") or "")
        raison_pred.append(result.get("raisonnement", "") or "")
        causes_ref.append(entry["cause"])
        raison_ref.append(entry["raisonnement"])

        if (i + 1) % 20 == 0:
            print(f"   {i+1}/{len(test_data)} — JSON valides : {n_valides}/{i+1}")

    return evaluer(
        nom, causes_pred, raison_pred, causes_ref, raison_ref,
        n_valides, len(test_data),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  6. AFFICHAGE
# ─────────────────────────────────────────────────────────────────────────────
def afficher(resultats: list[dict]):
    print(f"\n{'='*80}")
    print("RÉSULTATS COMPARATIFS")
    print(f"{'='*80}")
    print(
        f"{'Modèle':<30} {'JSON%':>6} "
        f"{'ROUGE cause':>12} {'ROUGE raison':>13} "
        f"{'BERT cause':>11} {'BERT raison':>12}"
    )
    print("-" * 80)
    for r in resultats:
        print(
            f"{r['nom']:<30} {r['json_valid']:>6.2%} "
            f"{r['rouge_cause']:>12.4f} {r['rouge_raison']:>13.4f} "
            f"{r['bert_cause']:>11.4f} {r['bert_raison']:>12.4f}"
        )
    print(f"{'='*80}")
    print("""
Lecture des métriques :
  JSON%        → % de réponses parseable en JSON (fiabilité du format)
  ROUGE-L      → chevauchement de mots exacts avec la référence (0→1)
  BERTScore    → similarité sémantique avec la référence (0→1)
  cause        → évaluation sur la phrase de cause uniquement
  raison       → évaluation sur le raisonnement complet en 3 étapes
""")


# ─────────────────────────────────────────────────────────────────────────────
#  7. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None,
                        help="Nombre d'exemples à évaluer (défaut: tous)")
    args = parser.parse_args()

    # ── Données ──────────────────────────────────────────────────────────────
    print("📥 Chargement des données...")
    test_data = charger_test(TEST_JSON, n=args.n)

    causes_ref  = [d["cause"]        for d in test_data]
    raison_ref  = [d["raisonnement"] for d in test_data]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "resultats_evaluation.json")

    # Reprise : si un run précédent a déjà produit des résultats partiels
    # (ex: crash pendant le chargement du 3e modèle), on les recharge et on
    # ne recalcule pas ce qui est déjà fait.
    resultats = []
    noms_deja_faits = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            resultats = json.load(f)
        noms_deja_faits = {r["nom"] for r in resultats}
        if noms_deja_faits:
            print(f"↩️  Reprise : {len(noms_deja_faits)} résultat(s) déjà calculé(s), conservés : {sorted(noms_deja_faits)}")

    def sauvegarder():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(resultats, f, ensure_ascii=False, indent=2)
        print(f"   📄 Sauvegardé dans {out_path}")

    # ── Baseline 1 : cause fréquente ─────────────────────────────────────────
    if "Baseline (cause fréquente)" not in noms_deja_faits:
        print("\n🔄 Baseline : cause la plus fréquente...")
        cause_freq = cause_la_plus_frequente(TRAIN_JSON)
        print(f"   Cause baseline : '{cause_freq}'")

        causes_baseline = [cause_freq] * len(test_data)
        raison_baseline = [""] * len(test_data)

        r_baseline = evaluer(
            "Baseline (cause fréquente)",
            causes_baseline, raison_baseline,
            causes_ref, raison_ref,
            len(test_data), len(test_data),
        )
        resultats.append(r_baseline)
        sauvegarder()
    else:
        print("\n⏭️  Baseline déjà calculée, on saute.")

    # ── Baseline 2 : zero-shot ────────────────────────────────────────────────
    if "Qwen2.5-1.5B zero-shot" not in noms_deja_faits:
        model_zs, tok_zs = charger_modele(lora_dir=None)
        r_zs = evaluer_modele("Qwen2.5-1.5B zero-shot", model_zs, tok_zs, test_data)
        resultats.append(r_zs)
        sauvegarder()
        del model_zs
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    else:
        print("\n⏭️  Zero-shot déjà calculé, on saute.")

    # ── Modèle fine-tuné ─────────────────────────────────────────────────────
    if "Qwen2.5-1.5B fine-tuné (toi)" not in noms_deja_faits:
        model_ft, tok_ft = charger_modele(lora_dir=LORA_DIR)
        r_ft = evaluer_modele("Qwen2.5-1.5B fine-tuné (toi)", model_ft, tok_ft, test_data)
        resultats.append(r_ft)
        sauvegarder()
    else:
        print("\n⏭️  Fine-tuné déjà calculé, on saute.")

    # ── Résultats ────────────────────────────────────────────────────────────
    afficher(resultats)


if __name__ == "__main__":
    main()
