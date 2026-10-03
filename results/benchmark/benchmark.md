# Inference benchmark (Tesla T4, 64 test lines, fp16, LoRA merged)

| Setup | s / line | lines / s | tokens / s | USD per 1k lines |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B + LoRA, batch 1 | 2.918 | 0.34 | 30 | 0.2837 |
| Qwen2.5-1.5B + LoRA, batch 16 | 0.251 | 3.99 | 344 | 0.0244 |
| Llama 3.3-70B on Groq (estimate) | | | | 0.1675 |

Peak GPU memory: 3.38 GB. Average request: 160 tokens in, 92 tokens out.
Prices: GPU 0.35 $/h, Groq 0.59 / 0.79 $ per 1M tokens in / out.
