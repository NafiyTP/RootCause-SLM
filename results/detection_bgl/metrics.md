# BGL detection (4,713,493 lines, 3,619 windows of 60 min)

Chronological split, test = last 724 windows (18.6% anomalous). 13603 templates in training, 412 test windows contain an unseen one.

| Detector | Precision | Recall | F1 | Flagged |
|---|---:|---:|---:|---:|
| Level rule (FATAL/FAILURE/SEVERE/ERROR) | 0.734 | 1.000 | 0.846 | 184 |
| PCA (unsupervised) | 0.203 | 0.541 | 0.295 | 359 |
| Logistic regression | 0.972 | 0.518 | 0.676 | 72 |
