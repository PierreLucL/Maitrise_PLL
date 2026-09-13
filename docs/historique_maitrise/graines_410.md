# Souris 410 : trois graines à 100 passages

## Contrôles

Les nouvelles sauvegardes ont été vérifiées par SHA-256 contre Narval. Valeurs finies, dimensions, partition des unités et temps vérifiés. Pour les comparaisons 410 : mêmes Adata, masque, régions, temps, paramètres hors graine et empreintes du code ; J0 et inputWN différents. Ces contrôles ne garantissent pas une pile numérique ou un matériel identique sur les nœuds. Les trois modèles utilisent tous l’ancien indice tt.

## Reconstruction

| Souris | Graine | pVar recalculée |
|---|---|---:|
| 410 | 2026 | 0.945365 |
| 410 | 2027 | 0.946359 |
| 410 | 2028 | 0.942939 |
| 415 | 2026 | 0.914296 |

## Résumé des 36 blocs par comparaison

| Graines | Pearson signé médian | Pearson norme PCA10 médian | CCA1 médiane | Moyenne des 10 CCA, médiane | Ratio norme B/A min–max | Pearson J médian | KS médian |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026 / 2027 | 0.597 | 0.899 | 0.987 | 0.715 | 0.71–2.30 | 0.181 | 0.0148 |
| 2026 / 2028 | 0.336 | 0.855 | 0.987 | 0.706 | 0.48–1.49 | 0.174 | 0.0150 |
| 2027 / 2028 | 0.065 | 0.848 | 0.986 | 0.715 | 0.35–1.39 | 0.182 | 0.0244 |

## Portée

Chaque paire source-cible conserve une matrice unités × temps. La PCA est centrée temporellement par unité, calculée exactement et limitée à dix composantes non blanchies ; la norme est celle des scores. La CCA est calculée par QR/SVD sur cette même session, sans contrôle mélangé ni évaluation indépendante. Il s’agit de notre implémentation de la description Neuron, pas d’une reproduction vérifiée ligne par ligne des scripts des auteurs.

La première CCA mesure le meilleur alignement et ignore certaines transformations de signe et d’échelle. Les rapports d’amplitude doivent être examinés séparément. Les 36 blocs et les trois comparaisons ne sont pas des répétitions indépendantes : chaque réseau et chaque région intervient plusieurs fois. KS est utilisé comme distance descriptive, pas comme test d’équivalence. Trois graines permettent un premier bilan, pas une preuve de validité biologique.

Le premier modèle de 415 a été contrôlé, mais sa stabilité entre graines ne peut pas encore être évaluée avec ce seul résultat.
