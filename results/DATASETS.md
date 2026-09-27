# Jeux de données

Relevé le 2026-09-27. Les fichiers eux-mêmes ne sont pas versionnés (`data/` est dans `.gitignore`).

| Fichier | Source | Taille complète | Utilisé ici |
|---|---|---|---|
| `pretrain_t2t_mini.jsonl` | [huggingface.co/datasets/jingyaogong/minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main) (miroir : ModelScope `gongjy/minimind_dataset`) | 1,24 Go | **Mac** : les 33 553 139 premiers octets (requête HTTP `Range: bytes=0-33554431`, coupés à la dernière ligne complète), soit **45 136 exemples** → `data/pretrain_t2t_mini.head32mb.jsonl`. **Kaggle** : fichier complet. |
| `sft_t2t_mini.jsonl` | même dépôt | 1,74 Go | **Kaggle** seulement (T7.3). |

## Licence

- **Code** de minimind : Apache-2.0 (`upstream/LICENSE`).
- **Jeu de données** : la fiche Hugging Face déclare **`apache-2.0` et `cc-by-nc-2.0`**. La fiche précise que les
  données agrègent des sources tierces (匠数, Magpie-Align, R1-Distill-SFT, COIG, Step-3.5-Flash-SFT…) et des données
  synthétiques, et qu'elles respectent « Apache-2.0, CC-BY-NC-2.0 等 ».
- Conséquence : on retient la plus restrictive, **CC-BY-NC-2.0 : usage non commercial, attribution obligatoire**.
  Un modèle de démonstration pour un portfolio est conforme ; aucun modèle entraîné ici ne doit être vendu ni servi
  commercialement.

## Format

Une ligne JSON par exemple, clé unique `text` (texte brut, majoritairement en chinois ; les dialogues sont déjà
aplatis). Chargé par `upstream/dataset/lm_dataset.py:PretrainDataset` via `datasets.load_dataset('json', …)`.
