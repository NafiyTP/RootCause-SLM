# Examples

## blk_-8388850594438929714: Anomaly, never_stored

```
Block blk_-8388850594438929714, 4 lines. Events in order of first appearance:
  2x INFO dfs.DataNode$DataXceiver: Receiving block <BLK> src: /<IP> dest: /<IP>
  1x INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: <PATH> <BLK>
  1x INFO dfs.DataNode$DataXceiver: writeBlock <BLK> received exception java.io.IOException: Could not read from stream
WARN or exception lines:
  081111 080012 24026 INFO dfs.DataNode$DataXceiver: writeBlock blk_-8388850594438929714 received exception java.io.IOException: Could not read from stream
A normal block always has: INFO dfs.DataNode$DataXceiver: Receiving block <BLK> src: /<IP> dest: /<IP>; INFO dfs.DataNode$PacketResponder: PacketResponder <N> for block <BLK> terminating; INFO dfs.DataNode$PacketResponder: Received block <BLK> of size <N> from /<IP>; INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: <IP> is added to <BLK> size <N>; INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: <PATH> <BLK>
```

- **rules**: anomalous=True; The block was allocated but never stored: the events a normal write ends with (addStoredBlock, PacketResponder terminating) never appear.. 
- **teacher**: anomalous=True; writeBlock exception. The block shows a writeBlock IOException and lacks the normal PacketResponder and addStoredBlock events, indicating a failure during block write.
- **v1**: anomalous=True; Erreur d'allocation de bloc dans HDFS. Étape 1 : Le log HDFS indique une erreur d'allocation de bloc. Étape 2 : Le message d'erreur BLOCK* NameSystem.allocateBlock suggère un problème avec l'allocation de blocs dans le système de fichiers. Étape 3 : Cela peut être dû à une erreur de configuration ou à une erreur de système de fichiers
- **base**: (no valid answer)
- **v2**: anomalous=True; writeBlock IOException. The block log contains a writeBlock IOException and a warning about failing to read the block, which is not part of normal block lifecycle.

## blk_8142509918875794502: Normal, re_replication

```
Block blk_8142509918875794502, 19 lines. Events in order of first appearance:
  4x INFO dfs.DataNode$DataXceiver: Receiving block <BLK> src: /<IP> dest: /<IP>
  1x INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: <PATH> <BLK>
  3x INFO dfs.DataNode$PacketResponder: PacketResponder <N> for block <BLK> terminating
  3x INFO dfs.DataNode$PacketResponder: Received block <BLK> of size <N> from /<IP>
  4x INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: <IP> is added to <BLK> size <N>
  1x INFO dfs.FSNamesystem: BLOCK* ask <IP> to replicate <BLK> to datanode(s) <IP>
  1x INFO dfs.DataNode: <IP> Starting thread to transfer block <BLK> to <IP>
  1x INFO dfs.DataNode$DataTransfer: <IP>:Transmitted block <BLK> to /<IP>
  1x INFO dfs.DataNode$DataXceiver: Received block <BLK> src: /<IP> dest: /<IP> of size <N>
A normal block always has: INFO dfs.DataNode$DataXceiver: Receiving block <BLK> src: /<IP> dest: /<IP>; INFO dfs.DataNode$PacketResponder: PacketResponder <N> for block <BLK> terminating; INFO dfs.DataNode$PacketResponder: Received block <BLK> of size <N> from /<IP>; INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: <IP> is added to <BLK> size <N>; INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: <PATH> <BLK>
```

- **rules**: anomalous=True; The NameNode had to copy the block to another DataNode (re-replication), which usually means a replica was lost or missing.. 
- **teacher**: anomalous=False; no issue. All expected normal events are present and there are no warning or exception lines, indicating no problem with this block.
- **v1**: anomalous=True; Tentative de replication du bloc de données. Étape 1 : Le système de fichiers HDFS a tenté de replicer le bloc de données blk_8142509918875794502. Étape 2 : Le processus s'est tenu à l'adresse IP 10.251.199.150 sur le port 50010. Étape 3 : Cependant, la tentative de replication a été anormalement déjouée par le système de fichiers.
- **base**: (no valid answer)
- **v2**: anomalous=False; none. All required normal events are present and there are no warning or exception lines, indicating no issue.

## blk_-8073339016342463696: Normal, none

```
Block blk_-8073339016342463696, 13 lines. Events in order of first appearance:
  3x INFO dfs.DataNode$DataXceiver: Receiving block <BLK> src: /<IP> dest: /<IP>
  1x INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: <PATH> <BLK>
  3x INFO dfs.DataNode$PacketResponder: PacketResponder <N> for block <BLK> terminating
  3x INFO dfs.DataNode$PacketResponder: Received block <BLK> of size <N> from /<IP>
  3x INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: <IP> is added to <BLK> size <N>
A normal block always has: INFO dfs.DataNode$DataXceiver: Receiving block <BLK> src: /<IP> dest: /<IP>; INFO dfs.DataNode$PacketResponder: PacketResponder <N> for block <BLK> terminating; INFO dfs.DataNode$PacketResponder: Received block <BLK> of size <N> from /<IP>; INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: <IP> is added to <BLK> size <N>; INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: <PATH> <BLK>
```

- **rules**: anomalous=True; Some events that normal blocks have are absent.. 
- **teacher**: anomalous=False; none. All expected normal events are present and there are no warning or exception lines, indicating no problem with this block.
- **v1**: anomalous=True; Erreur d'allocation de bloc dans HDFS. Étape 1 : Le log HDFS indique une erreur d'allocation de bloc. Étape 2 : Le message d'erreur BLOCK* NameSystem.allocateBlock suggère un problème avec l'allocation de blocs dans le système de fichiers. Étape 3 : Cela peut être dû à une erreur de configuration ou à une erreur de système de fichiers
- **base**: (no valid answer)
- **v2**: anomalous=False; no issue. All expected events for a normal block appear and there are no warning or exception lines, indicating no problem.

## blk_1186541281926863677: Anomaly, empty_packet

```
Block blk_1186541281926863677, 16 lines. Events in order of first appearance:
  3x INFO dfs.DataNode$DataXceiver: Receiving block <BLK> src: /<IP> dest: /<IP>
  1x INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: <PATH> <BLK>
  3x INFO dfs.DataNode$BlockReceiver: Receiving empty packet for block <BLK>
  3x INFO dfs.DataNode$PacketResponder: PacketResponder <N> for block <BLK> terminating
  3x INFO dfs.DataNode$PacketResponder: Received block <BLK> of size <N> from /<IP>
  3x INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: <IP> is added to <BLK> size <N>
A normal block always has: INFO dfs.DataNode$DataXceiver: Receiving block <BLK> src: /<IP> dest: /<IP>; INFO dfs.DataNode$PacketResponder: PacketResponder <N> for block <BLK> terminating; INFO dfs.DataNode$PacketResponder: Received block <BLK> of size <N> from /<IP>; INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: <IP> is added to <BLK> size <N>; INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: <PATH> <BLK>
```

- **rules**: anomalous=True; A DataNode received an empty packet while writing the block.. 
- **teacher**: anomalous=False; no issue. All required normal events are present and there are no warning or exception lines, indicating no problem.
- **v1**: anomalous=True; Détection d'un paquet vide dans le répertoire de blocs. Étape 1 : Le message indique la détention d'une opération de réception de blocs. Étape 2 : L'opération consiste à recevoir un bloc spécifique. Étape 3 : Le message suggère que le contenu du bloc est vide, ce qui peut être une anomalie dans les opérations de réception de blocs.
- **base**: (no valid answer)
- **v2**: anomalous=False; no issue. All expected events for a normal block appear and there are no warning or exception lines, indicating no problem.

## blk_7420882489167329474: Anomaly, failed_delete

```
Block blk_7420882489167329474, 20 lines. Events in order of first appearance:
  3x INFO dfs.DataNode$DataXceiver: Receiving block <BLK> src: /<IP> dest: /<IP>
  1x INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: <PATH> <BLK>
  3x INFO dfs.DataNode$PacketResponder: PacketResponder <N> for block <BLK> terminating
  3x INFO dfs.DataNode$PacketResponder: Received block <BLK> of size <N> from /<IP>
  3x INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: <IP> is added to <BLK> size <N>
  3x INFO dfs.FSNamesystem: BLOCK* NameSystem.delete: <BLK> is added to invalidSet of <IP>
  3x INFO dfs.FSDataset: Deleting block <BLK> file <PATH><BLK>
  1x WARN dfs.FSDataset: Unexpected error trying to delete block <BLK>. BlockInfo not found in volumeMap.
WARN or exception lines:
  081111 090203 19 WARN dfs.FSDataset: Unexpected error trying to delete block blk_7420882489167329474. BlockInfo not found in volumeMap.
A normal block always has: INFO dfs.DataNode$DataXceiver: Receiving block <BLK> src: /<IP> dest: /<IP>; INFO dfs.DataNode$PacketResponder: PacketResponder <N> for block <BLK> terminating; INFO dfs.DataNode$PacketResponder: Received block <BLK> of size <N> from /<IP>; INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: <IP> is added to <BLK> size <N>; INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: <PATH> <BLK>
```

- **rules**: anomalous=True; A delete failed: the DataNode had no record of the block (BlockInfo not found).. 
- **teacher**: anomalous=True; Block deletion error. The block shows a WARN about failing to delete it because its BlockInfo is missing, indicating a deletion problem not present in normal block lifecycles.
- **v1**: anomalous=True; Bloc supprimé dans le volume de données HDFS erroné. Étape 1 : Le système HDFS a tenté de supprimer un bloc erreur. Étape 2 : Le bloc a été supprimé avec succès. Étape 3 : Le système a généré un message d'erreur pour signaler l'anomalie
- **base**: anomalous=False; Unexpected error. The presence of a WARN line indicating an unexpected error during block deletion suggests potential issues with data integrity or storage management.
- **v2**: anomalous=True; Delete error. The block shows a WARN about failing to delete it because its BlockInfo was missing, which is not part of normal block lifecycle.

## blk_-8168561708838519032: Anomaly, redundant

```
Block blk_-8168561708838519032, 14 lines. Events in order of first appearance:
  3x INFO dfs.DataNode$DataXceiver: Receiving block <BLK> src: /<IP> dest: /<IP>
  1x INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: <PATH> <BLK>
  3x INFO dfs.DataNode$PacketResponder: PacketResponder <N> for block <BLK> terminating
  3x INFO dfs.DataNode$PacketResponder: Received block <BLK> of size <N> from /<IP>
  3x INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: <IP> is added to <BLK> size <N>
  1x WARN dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: Redundant addStoredBlock request received for <BLK> on <IP> size <N>
WARN or exception lines:
  081111 090946 27 WARN dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: Redundant addStoredBlock request received for blk_-8168561708838519032 on 10.251.109.209:50010 size 3543649
A normal block always has: INFO dfs.DataNode$DataXceiver: Receiving block <BLK> src: /<IP> dest: /<IP>; INFO dfs.DataNode$PacketResponder: PacketResponder <N> for block <BLK> terminating; INFO dfs.DataNode$PacketResponder: Received block <BLK> of size <N> from /<IP>; INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: <IP> is added to <BLK> size <N>; INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: <PATH> <BLK>
```

- **rules**: anomalous=True; The same replica was reported as stored twice (redundant addStoredBlock).. 
- **teacher**: anomalous=True; Redundant addStoredBlock. The block shows a WARN about a redundant addStoredBlock request, which is not part of normal block processing, indicating a potential duplication issue.
- **v1**: anomalous=True; Réception d'une demande de ajout de bloc répétée. Étape 1 : La FSNamesystem.readdfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock est appelée pour ajouter un bloc dans le système de fichiers. Étape 2 : Le addStoredBlock est appelé avec la requête spécifiée, ce qui indique une réception de demandes de ajout de bloc. Étape 3 : La redondance du bloc est vérifiée avant d'être ajouté au système de fichiers.
- **base**: anomalous=False; Redundant addStoredBlock request. The presence of redundant addStoredBlock requests indicates that there might be unnecessary data replication or synchronization issues within the Hadoop Distributed File System (HDFS) cluster.
- **v2**: anomalous=True; Redundant addStoredBlock warning. The block log contains a WARN about a redundant addStoredBlock request, which is not part of normal block lifecycle, indicating an anomaly.

## blk_-1772502581387873802: Anomaly, no_file

```
Block blk_-1772502581387873802, 19 lines. Events in order of first appearance:
  3x INFO dfs.DataNode$DataXceiver: Receiving block <BLK> src: /<IP> dest: /<IP>
  1x INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: <PATH> <BLK>
  3x INFO dfs.DataNode$PacketResponder: PacketResponder <N> for block <BLK> terminating
  3x INFO dfs.DataNode$PacketResponder: Received block <BLK> of size <N> from /<IP>
  3x INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: addStoredBlock request received for <BLK> on <IP> size <N> But it does not belong to any file.
  3x INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: <IP> is added to <BLK> size <N>
  3x INFO dfs.FSDataset: Deleting block <BLK> file <PATH><BLK>
A normal block always has: INFO dfs.DataNode$DataXceiver: Receiving block <BLK> src: /<IP> dest: /<IP>; INFO dfs.DataNode$PacketResponder: PacketResponder <N> for block <BLK> terminating; INFO dfs.DataNode$PacketResponder: Received block <BLK> of size <N> from /<IP>; INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: <IP> is added to <BLK> size <N>; INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: <PATH> <BLK>
```

- **rules**: anomalous=True; The NameNode received addStoredBlock for a block that belongs to no file.. 
- **teacher**: anomalous=True; block not belonging to any file. The block was reported as not belonging to any file and was subsequently deleted, which is not part of normal block lifecycle.
- **v1**: anomalous=True; Bloc ajouté sans fichier associé. Étape 1 : Le bloc est ajouté au système de fichiers, Étape 2 : L'adresse IP et le port sont corrects, Étape 3 : La taille du bloc est valide
- **base**: (no valid answer)
- **v2**: anomalous=True; block deleted after allocation. The block was allocated but then immediately marked for deletion, which is abnormal.

## blk_610605540713393724: Anomaly, re_replication

```
Block blk_610605540713393724, 26 lines. Events in order of first appearance:
  4x INFO dfs.DataNode$DataXceiver: Receiving block <BLK> src: /<IP> dest: /<IP>
  1x INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: <PATH> <BLK>
  1x INFO dfs.FSNamesystem: BLOCK* ask <IP> to replicate <BLK> to datanode(s) <IP>
  3x INFO dfs.DataNode$PacketResponder: PacketResponder <N> for block <BLK> terminating
  3x INFO dfs.DataNode$PacketResponder: Received block <BLK> of size <N> from /<IP>
  4x INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: <IP> is added to <BLK> size <N>
  1x INFO dfs.DataNode: <IP> Starting thread to transfer block <BLK> to <IP>
  1x INFO dfs.DataNode$DataTransfer: <IP>:Transmitted block <BLK> to /<IP>
  1x INFO dfs.DataNode$DataXceiver: Received block <BLK> src: /<IP> dest: /<IP> of size <N>
  4x INFO dfs.FSDataset: Deleting block <BLK> file <PATH><BLK>
  3x INFO dfs.FSNamesystem: BLOCK* NameSystem.delete: <BLK> is added to invalidSet of <IP>
A normal block always has: INFO dfs.DataNode$DataXceiver: Receiving block <BLK> src: /<IP> dest: /<IP>; INFO dfs.DataNode$PacketResponder: PacketResponder <N> for block <BLK> terminating; INFO dfs.DataNode$PacketResponder: Received block <BLK> of size <N> from /<IP>; INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: <IP> is added to <BLK> size <N>; INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: <PATH> <BLK>
```

- **rules**: anomalous=True; The NameNode had to copy the block to another DataNode (re-replication), which usually means a replica was lost or missing.. 
- **teacher**: anomalous=True; block deletion after replication. The block shows normal replication events but also includes deletion and invalidation events, indicating it was removed unexpectedly after being stored.
- **v1**: anomalous=True; Bloc de données corrompu. Étape 1 : Le système de fichiers HDFS a détecté un bloc de données corrompu. Étape 2 : Le bloc a été ajouté à l'ensemble des blocs invalides. Étape 3 : Le système tentera de répliquer le bloc pour assurer la cohérence des données.
- **base**: (no valid answer)
- **v2**: anomalous=True; block deletion. The block log contains delete and invalidation events which are not part of a typical block lifecycle, indicating the block was removed after being stored.

## blk_6180515937140679205: Anomaly, replication_timeout

```
Block blk_6180515937140679205, 17 lines. Events in order of first appearance:
  3x INFO dfs.DataNode$DataXceiver: Receiving block <BLK> src: /<IP> dest: /<IP>
  1x INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: <PATH> <BLK>
  3x INFO dfs.DataNode$PacketResponder: PacketResponder <N> for block <BLK> terminating
  3x INFO dfs.DataNode$PacketResponder: Received block <BLK> of size <N> from /<IP>
  4x INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: <IP> is added to <BLK> size <N>
  1x INFO dfs.FSNamesystem: BLOCK* ask <IP> to replicate <BLK> to datanode(s) <IP>
  1x INFO dfs.DataNode: <IP> Starting thread to transfer block <BLK> to <IP>
  1x WARN dfs.PendingReplicationBlocks$PendingReplicationMonitor: PendingReplicationMonitor timed out block <BLK>
WARN or exception lines:
  081111 092433 16 WARN dfs.PendingReplicationBlocks$PendingReplicationMonitor: PendingReplicationMonitor timed out block blk_6180515937140679205
A normal block always has: INFO dfs.DataNode$DataXceiver: Receiving block <BLK> src: /<IP> dest: /<IP>; INFO dfs.DataNode$PacketResponder: PacketResponder <N> for block <BLK> terminating; INFO dfs.DataNode$PacketResponder: Received block <BLK> of size <N> from /<IP>; INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: <IP> is added to <BLK> size <N>; INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: <PATH> <BLK>
```

- **rules**: anomalous=True; A pending replication of the block timed out.. 
- **teacher**: anomalous=True; replication timeout. The block shows a pending replication monitor timeout warning, which is not part of normal block lifecycle, indicating a replication problem.
- **v1**: anomalous=True; Bloc de données HDFS timeout. Étape 1 : Le bloc de données a été en attente pour plus de 24 heures. Étape 2 : Le processus PendingReplicationMonitor n'a pas pu gérer le bloc de données. Étape 3 : Le temps de vie du bloc de données a été écoulé
- **base**: anomalous=True; Timed-out pending replication. The presence of a WARN message indicating a timeout during pending replication suggests that the block was not successfully replicated as expected.
- **v2**: anomalous=True; replication timeout. The block shows a pending replication monitor timeout warning, which is not part of normal block lifecycle, indicating a replication issue.

