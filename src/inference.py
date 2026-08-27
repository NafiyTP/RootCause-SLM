"""
inference.py — Inférence avec le modèle fine-tuné
 
Charge Qwen2.5-1.5B + les adaptateurs LoRA et génère
une analyse JSON pour n'importe quel log HDFS.
 
Usage :
    python inference.py
    python inference.py --log "081109 203615 148 WARN dfs.DataNode: Got exception..."
    python inference.py --file logs_a_analyser.txt
"""
 
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
 
# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
import os

BASE_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
LORA_DIR    = os.path.join(BASE_DIR, "..", "modele_hdfs")
MAX_NEW_TOKENS = 300
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
 
SYSTEM_PROMPT = (
    "Tu es un expert en systèmes distribués Hadoop/HDFS. "
    "Étant donné un log, tu identifies s'il s'agit d'une anomalie ou d'une opération normale, "
    "tu fournis une cause technique précise et un raisonnement en 3 étapes."
)
 
# Quelques logs d'exemple pour tester sans fichier externe
EXEMPLES = [
    "081109 203615 148 WARN dfs.DataNode$DataXceiver: Got exception while serving blk_-6952295868487656571 to /10.251.73.220",
    "081109 204005 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.73.220:50010 is added to blk_7128370237687728475",
    "081109 204525 512 INFO dfs.DataNode$PacketResponder: PacketResponder 2 for block blk_572492839287299681 terminating",
]
 
# ─────────────────────────────────────────────────────────────────────────────
#  1. CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
def charger_modele():
    print(f"📥 Chargement du modèle de base : {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

 
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map={"": DEVICE},
    )
 
    print(f"📥 Chargement des adaptateurs LoRA : {LORA_DIR}")
    model = PeftModel.from_pretrained(model, LORA_DIR)
    model.eval()
 
    print(f"✅ Modèle prêt sur {DEVICE}\n")
    return model, tokenizer
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  2. INFÉRENCE SUR UN LOG
# ─────────────────────────────────────────────────────────────────────────────
def analyser_log(log: str, model, tokenizer) -> dict:
    """
    Prend une ligne de log brute, retourne un dict avec :
    - prediction  : "Normal" ou "Anomaly"
    - cause       : cause technique en une phrase
    - raisonnement : explication en 3 étapes
    - raw         : texte brut généré par le modèle
    """
    # Format ChatML — identique à ce qu'on a utilisé pendant le training
    # Le modèle a été entraîné sans label en entrée → il doit prédire lui-même
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Analyse ce log HDFS et détermine s'il s'agit d'une anomalie ou d'une "
                    f"opération normale. Réponds en JSON pur sans backticks :\n"
                    f'{{"prediction": "Normal" ou "Anomaly", '
                    f'"cause": "...", "raisonnement": "Étape 1 : ... Étape 2 : ... Étape 3 : ..."}}\n\n'
                    f"Log : {log[:300]}"
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
            temperature=0.1,       # quasi-déterministe
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
 
    # Décoder uniquement les tokens générés (pas le prompt)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    texte = tokenizer.decode(generated, skip_special_tokens=True).strip()
 
    # Parser le JSON
    try:
        import re
        texte_clean = re.sub(r"^```(?:json)?", "", texte).strip().rstrip("```").strip()
        resultat = json.loads(texte_clean)
        resultat["raw"] = texte
        return resultat
    except json.JSONDecodeError:
        return {
            "prediction":   "Parse error",
            "cause":        texte[:200],
            "raisonnement": "",
            "raw":          texte,
        }
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  3. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",  type=str, help="Log à analyser")
    parser.add_argument("--file", type=str, help="Fichier texte, un log par ligne")
    args = parser.parse_args()
 
    model, tokenizer = charger_modele()
 
    if args.log:
        logs = [args.log]
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            logs = [l.strip() for l in f if l.strip()]
    else:
        print("Aucun log fourni — utilisation des exemples intégrés\n")
        logs = EXEMPLES
 
    print("=" * 72)
    for i, log in enumerate(logs):
        print(f"\n[Log {i+1}]")
        print(f"  Entrée : {log[:100]}...")
 
        resultat = analyser_log(log, model, tokenizer)
 
        print(f"  Prédiction   : {resultat.get('prediction', '?')}")
        print(f"  Cause        : {resultat.get('cause', '?')}")
        print(f"  Raisonnement : {resultat.get('raisonnement', '?')[:200]}")
        print("-" * 72)
 
 
if __name__ == "__main__":
    main()