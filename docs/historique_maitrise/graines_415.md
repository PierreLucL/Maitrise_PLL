# Comparaison des trois graines : souris 415, 100 passages

Les deux nouveaux fichiers ont été vérifiés par SHA-256 contre Narval. Valeurs numériques finies, dimensions, temps et partition des unités contrôlés. Les comparaisons vérifient mêmes données Adata, temps, masque, régions, paramètres hors seed et empreintes des sources ; J0 et inputWN différents. Le matériel et la totalité de la pile numérique des nœuds ne sont pas contrôlés. Tous les entraînements utilisent la même version historique avec indice tt.

## Reconstruction

| Graine | pVar recalculée |
|---|---:|
| 2026 | 0.914296 |
| 2027 | 0.916001 |
| 2028 | 0.919666 |

## Comparaison des 36 contributions

| Graines | Pearson norme PCA10 médian | CCA1 médiane | Moyenne des 10 CCA, médiane | Ratio norme B/A min–max | Pearson signé médian | Pearson J médian | KS J médian |
|---|---:|---:|---:|---|---:|---:|---:|
| 2026/2027 | 0.903 | 0.988 | 0.755 | 0.46–1.17 | 0.620 | 0.151 | 0.0169 |
| 2026/2028 | 0.908 | 0.989 | 0.755 | 0.55–1.56 | 0.609 | 0.101 | 0.0210 |
| 2027/2028 | 0.917 | 0.989 | 0.774 | 0.67–1.72 | 0.639 | 0.148 | 0.0155 |

## Méthode et limites

Calculs identiques à ceux appliqués à 410. Courants complets J[cible,source] @ R[source] ; PCA exacte, centrage temporel de chaque unité, dix scores non blanchis, norme euclidienne ; CCA par QR/SVD. Il s’agit de notre implémentation de la description Neuron, pas d’une vérification ligne à ligne des scripts des auteurs. Les sommes signées restent un diagnostic complémentaire.

La CCA est ajustée et évaluée sur la même session, sans contrôle mélangé ni validation indépendante. La première CCA ne caractérise pas toutes les dimensions et ne garantit ni amplitudes ni signes identiques. Les trois paires réutilisent les mêmes réseaux : ni ces paires ni les 36 blocs ne constituent des observations indépendantes. KS est une distance descriptive. Des distributions de J proches ne démontrent pas la validité biologique des interactions.

## Comparaison descriptive à 410

| Souris | Étendue des médianes Pearson norme PCA10 sur les trois paires | Étendue des médianes CCA1 |
|---|---|---|
| 410 | 0.848–0.899 | 0.986–0.987 |
| 415 | 0.903–0.917 | 0.988–0.989 |
