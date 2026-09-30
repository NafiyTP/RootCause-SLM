# Results (n = 527)

| System | json_strict | json_lenient | cause_exact | rouge_cause | rouge_reason | bert_cause | bert_reason |
|---|---:|---:|---:|---:|---:|---:|---:|
| Most frequent cause | 1.000 | 1.000 | 0.461 | 0.594 | 0.347 | 0.862 | 0.797 |
| Label majority | 1.000 | 1.000 | 0.461 | 0.589 | 0.356 | 0.869 | 0.783 |
| Template retrieval (kNN) | 1.000 | 1.000 | 0.548 | 0.860 | 0.456 | 0.953 | 0.825 |
| Qwen2.5-1.5B zero-shot | 0.065 | 0.065 | 0.000 | 0.008 | 0.017 | 0.049 | 0.048 |
| Qwen2.5-1.5B few-shot (3) | 1.000 | 1.000 | 0.000 | 0.393 | 0.292 | 0.792 | 0.778 |
| Qwen2.5-1.5B + LoRA (ours) | 1.000 | 1.000 | 0.617 | 0.874 | 0.506 | 0.958 | 0.836 |

## By label

| System | ROUGE-L reasoning, Normal | ROUGE-L reasoning, Anomaly | cause exact match, Normal | cause exact match, Anomaly |
|---|---:|---:|---:|---:|
| Most frequent cause | 0.438 | 0.247 | 0.877 | 0.000 |
| Label majority | 0.438 | 0.266 | 0.877 | 0.000 |
| Template retrieval (kNN) | 0.451 | 0.461 | 0.917 | 0.140 |
| Qwen2.5-1.5B zero-shot | 0.000 | 0.035 | 0.000 | 0.000 |
| Qwen2.5-1.5B few-shot (3) | 0.327 | 0.254 | 0.000 | 0.000 |
| Qwen2.5-1.5B + LoRA (ours) | 0.552 | 0.455 | 0.917 | 0.284 |

## ROUGE-L reasoning, bootstrap 95% CI

- Most frequent cause: 0.347 [0.338, 0.357]
- Label majority: 0.356 [0.347, 0.366]
- Template retrieval (kNN): 0.456 [0.443, 0.467]
- Qwen2.5-1.5B zero-shot: 0.017 [0.011, 0.023]
- Qwen2.5-1.5B few-shot (3): 0.292 [0.286, 0.299]
- Qwen2.5-1.5B + LoRA (ours): 0.506 [0.493, 0.519]
