"""
PyTorch dataset and collator for fine-tuning Qwen2.5 on annotated HDFS logs.

Each example is one log line + its label as the prompt, and the teacher's
{"cause", "raisonnement"} JSON as the target. The loss is only computed on the target.

Quick check:
    python src/dataset.py
"""

import json
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# The annotations are in French, so the prompt is too.
SYSTEM_PROMPT = (
    "Tu es un expert en systèmes distribués Hadoop/HDFS. "
    "Étant donné un log et son label (Normal ou Anomaly), "
    "tu fournis une cause technique précise et un raisonnement en 3 étapes."
)


def format_example(entry: dict) -> tuple[str, str]:
    """Split an entry into (prompt, response). I need them separately to build the loss mask."""
    prompt = (
        f"Log HDFS : {entry['log'][:300]}\n"
        f"Label : {entry['label']}"
    )
    response = json.dumps(
        {"cause": entry["cause"], "raisonnement": entry["raisonnement"]},
        ensure_ascii=False,
    )
    return prompt, response


class HDFSLogDataset(Dataset):
    """
    Tokenizes each example twice: the prompt alone (to know its length) and
    prompt + response in ChatML. Labels are a copy of input_ids with the prompt
    positions set to -100, so cross-entropy ignores them.
    """

    def __init__(self, json_path: str, tokenizer: Any, max_length: int = 512,
                 skip_empty: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)

        skipped = 0
        for entry in raw:
            # a few teacher calls returned invalid JSON and left empty fields
            if skip_empty and (not entry.get("cause") or not entry.get("raisonnement")):
                skipped += 1
                continue
            self.data.append(entry)

        print(f"Loaded {len(self.data)} examples ({skipped} skipped, empty annotation)")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        prompt, response = format_example(self.data[idx])

        full_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,  # adds "<|im_start|>assistant\n"
        )

        full = self.tokenizer(full_text, max_length=self.max_length, truncation=True,
                              padding=False, return_tensors="pt")
        prompt_ids = self.tokenizer(prompt_text, max_length=self.max_length, truncation=True,
                                    padding=False, return_tensors="pt")

        input_ids = full["input_ids"].squeeze(0)
        attention_mask = full["attention_mask"].squeeze(0)
        n_prompt = prompt_ids["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:n_prompt] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass
class HDFSDataCollator:
    """
    Pads each batch to its longest sequence (not to max_length), on the right.
    Padding gets pad_token_id in input_ids, 0 in the attention mask and -100 in labels.
    """

    tokenizer: Any

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(e["input_ids"].shape[0] for e in examples)

        input_ids, attention_mask, labels = [], [], []
        for e in examples:
            n_pad = max_len - e["input_ids"].shape[0]
            input_ids.append(torch.cat([e["input_ids"],
                                        torch.full((n_pad,), self.tokenizer.pad_token_id)]))
            attention_mask.append(torch.cat([e["attention_mask"],
                                             torch.zeros(n_pad, dtype=torch.long)]))
            labels.append(torch.cat([e["labels"], torch.full((n_pad,), -100)]))

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "data", "hdfs_dataset.json")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = HDFSLogDataset(json_path, tokenizer, max_length=512)
    loader = DataLoader(dataset, batch_size=4, shuffle=True,
                        collate_fn=HDFSDataCollator(tokenizer))
    batch = next(iter(loader))

    print("input_ids     ", tuple(batch["input_ids"].shape))
    print("attention_mask", tuple(batch["attention_mask"].shape))
    print("labels        ", tuple(batch["labels"].shape))

    first = batch["labels"][0]
    print(f"example 0: {(first == -100).sum().item()} masked tokens, "
          f"{(first != -100).sum().item()} supervised")
    print("decoded target:", tokenizer.decode(batch["input_ids"][0][first != -100]))
