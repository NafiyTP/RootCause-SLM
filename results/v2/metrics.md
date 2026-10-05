# v2 explanations, 123 test blocks (83 anomalous, 40 normal)

Scored against the raw log (gold = what is wrong in the whole block), not against the teacher.

Anomalous blocks: correct cause + wrong or vague cause + missed (says no problem) = 100%.

| System | Valid JSON | Correct cause | Wrong or vague | Missed | Invents on normal (random) | Invents on normal (re-replication) | Evidence in input |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rules (pipeline sentence) | 100% | 100% | 0% | 0% | 60% | 100% | - |
| Teacher (large model) | 100% | 67% | 13% | 19% | 40% | 0% | 99% |
| v1 student (one line) | 99% | 39% | 61% | 0% | - | - | - |
| Qwen2.5-1.5B, no fine-tuning | 27% | 4% | 0% | 96% | 0% | 0% | 100% |
| v2 student (block summary) | 100% | 63% | 24% | 13% | 40% | 0% | 98% |

Rules define the gold categories, so their correct-cause score is 100% by construction;
the question is how close the language models get, and how often they invent.

## Correct cause per category

| Category | n | Rules (pipeline sentence) | Teacher (large model) | v1 student (one line) | Qwen2.5-1.5B, no fine-tuning | v2 student (block summary) |
|---|---:|---:|---:|---:|---:|---:|
| never_stored | 15 | 100% | 100% | 0% | 0% | 100% |
| failed_delete | 15 | 100% | 100% | 13% | 0% | 100% |
| no_file | 5 | 100% | 100% | 20% | 0% | 60% |
| redundant | 15 | 100% | 100% | 27% | 0% | 100% |
| empty_packet | 15 | 100% | 0% | 47% | 0% | 7% |
| replication_timeout | 3 | 100% | 100% | 100% | 100% | 100% |
| re_replication | 15 | 100% | 20% | 100% | 0% | 0% |
