# Block-level detection (100,000 lines, 7,940 blocks)

Chronological split, test = last 1,588 blocks (5.7% anomalies).

| Detector | Precision | Recall | F1 | Flagged |
|---|---:|---:|---:|---:|
| WARN/ERROR rule | 0.000 | 0.000 | 0.000 | 0 |
| PCA (unsupervised) | 0.144 | 0.725 | 0.240 | 458 |
| Logistic regression | 0.154 | 0.780 | 0.257 | 462 |

Detection throughput on CPU: about 36,102 lines/s (parsing included).

## Templates with the largest positive weight

- +2.59  `INFO dfs.DataNode$BlockReceiver: Receiving empty packet for block <BLK>`
- +1.78  `INFO dfs.FSDataset: Deleting block <BLK> file <PATH><BLK>`
- +1.41  `INFO dfs.DataNode: <IP> Starting thread to transfer block <BLK> to <IP>`
- +1.41  `INFO dfs.FSNamesystem: BLOCK* ask <IP> to replicate <BLK> to datanode(s) <IP>`
- +0.92  `INFO dfs.DataNode$DataTransfer: <IP>:Transmitted block <BLK> to /<IP>`
- +0.67  `INFO dfs.DataNode$DataXceiver: Received block <BLK> src: /<IP> dest: /<IP> of size <N>`
- +0.66  `INFO dfs.DataNode$DataXceiver: <IP> Served block <BLK> to /<IP>`
- +0.43  `INFO dfs.DataNode$DataXceiver: writeBlock <BLK> received exception java.io.IOException: Could not read from stream`
