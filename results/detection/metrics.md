# Block-level detection (11,175,629 lines, 575,061 blocks)

Chronological split, test = last 115,013 blocks (1.5% anomalies).

| Detector | Precision | Recall | F1 | Flagged |
|---|---:|---:|---:|---:|
| WARN/ERROR rule | 1.000 | 0.248 | 0.398 | 417 |
| PCA (unsupervised) | 0.379 | 0.051 | 0.090 | 227 |
| Logistic regression | 0.955 | 0.992 | 0.973 | 1744 |

Detection throughput on CPU: about 117,022 lines/s (parsing included).

## Templates with the largest positive weight

- +22.87  `WARN dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: Redundant addStoredBlock request received for <BLK> on <IP> size <N>`
- +21.84  `INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: addStoredBlock request received for <BLK> on <IP> size <N> But it does not belong to any file.`
- +19.15  `WARN dfs.FSDataset: Unexpected error trying to delete block <BLK>. BlockInfo not found in volumeMap.`
- +14.17  `INFO dfs.FSNamesystem: BLOCK* NameSystem.delete: <BLK> is added to invalidSet of <IP>`
- +9.77  `INFO dfs.DataNode$BlockReceiver: Receiving empty packet for block <BLK>`
- +8.19  `WARN dfs.PendingReplicationBlocks$PendingReplicationMonitor: PendingReplicationMonitor timed out block <BLK>`
- +7.68  `INFO dfs.DataNode: <IP> Starting thread to transfer block <BLK> to <IP>`
- +7.68  `INFO dfs.FSNamesystem: BLOCK* ask <IP> to replicate <BLK> to datanode(s) <IP>`
