# Block-level detection (104,815 lines, 7,940 blocks)

Chronological split, test = last 1,588 blocks (5.7% anomalies).

| Detector | Precision | Recall | F1 | Flagged |
|---|---:|---:|---:|---:|
| WARN/ERROR rule | 1.000 | 0.011 | 0.022 | 1 |
| PCA (unsupervised) | 0.980 | 0.549 | 0.704 | 51 |
| Logistic regression | 0.982 | 0.593 | 0.740 | 55 |

Detection throughput on CPU: about 102,124 lines/s (parsing included).

## Templates with the largest positive weight

- +2.58  `INFO dfs.DataNode$BlockReceiver: Receiving empty packet for block <BLK>`
- +1.97  `INFO dfs.FSDataset: Deleting block <BLK> file <PATH><BLK>`
- +1.50  `INFO dfs.DataNode: <IP> Starting thread to transfer block <BLK> to <IP>`
- +1.50  `INFO dfs.FSNamesystem: BLOCK* ask <IP> to replicate <BLK> to datanode(s) <IP>`
- +0.88  `INFO dfs.DataNode$DataTransfer: <IP>:Transmitted block <BLK> to /<IP>`
- +0.69  `INFO dfs.DataNode$DataXceiver: <IP> Served block <BLK> to /<IP>`
- +0.56  `INFO dfs.DataNode$DataXceiver: Received block <BLK> src: /<IP> dest: /<IP> of size <N>`
- +0.51  `INFO dfs.DataNode$DataXceiver: writeBlock <BLK> received exception java.io.IOException: Could not read from stream`
