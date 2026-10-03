# Explanation audit (250 anomalous test lines)

## What is actually wrong with the block

| Block-level evidence | Blocks |
|---|---:|
| write never completed | 101 (40%) |
| failed delete (BlockInfo not found) | 82 (33%) |
| addStoredBlock for a block of no file | 47 (19%) |
| extra replication | 16 (6%) |
| empty packet | 4 (2%) |

## Do the explanations mention it?

| System | Mentions the block-level evidence |
|---|---:|
| teacher | 1 / 250 |
| finetuned | 0 / 250 |

## Most frequent causes given

- teacher: Erreur d'allocation de bloc dans HDFS (166); Connexion réseau anormale (37); Connexion non autorisée (19); Erreur d'allocation de bloc (16); Connexion anormale au DataNode (4)
- finetuned: Erreur d'allocation de bloc (142); Connexion non autorisée (54); Erreur d'allocation de bloc dans HDFS (41); Connexion réseau anormale (8); Connexion entrante non autorisée (3)
