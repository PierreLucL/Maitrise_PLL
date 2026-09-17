# Spécificité après projection du signal partagé — 14 septembre 2026

## Méthode et portée

Six modèles historiques à 100 passages (410/415, graines 2026–2028). Le prédicteur commun est la moyenne à poids égaux des six moyennes régionales de Adata, interpolée linéairement sur tRNN. Il inclut les régions source et cible. Pour chaque unité cible de chaque courant J[cible,source] @ RNN[source], l’ordonnée et la pente de la projection sont ajustées uniquement pour tRNN < 230 s. Les coefficients restent figés ensuite. Aucun modèle n’est réentraîné et aucune GSR n’est ajoutée au prétraitement.

PCA exacte à dix composantes non blanchies, centrage et axes ajustés sur le même premier bloc. CCA ajustée sur ce bloc puis corrélations des dix paires canoniques évaluées après 250 s, sans réordonner les dimensions ni réorienter leurs signes au test. La moyenne des dimensions 2–10 est complémentaire de CCA1. Pour une cible et une source dans la graine A, la même source dans B doit dépasser chacune des cinq autres sources pour gagner. Les trois paires de graines sont 2026/2027, 2026/2028 et 2027/2028, dans cet ordre.

108 comparaisons dépendantes par souris et par mode. Ce sont des descriptions, pas des répétitions biologiques indépendantes ni des p-values. Le RNN a appris toute la session : seul le transfert temporel des transformations est évalué. La composante projetée peut contenir un signal biologique et le résidu ne constitue pas une vérité terrain régionale. Le contrôle original est recalculé dans la même implémentation pour isoler l’effet de projection ; les décalages temporels historiques ne sont pas recalculés.

## Résultats

| Souris | Mode | CCA1 médiane test | Bonne source CCA1 | Marge CCA1 médiane | CCA2–10 médiane | Bonne source CCA2–10 | Bonne source norme |
|---|---|---:|---:|---:|---:|---:|---:|
| 410 | Courants originaux | 0.982 | 89/108 | +0.0022 | 0.646 | 108/108 | 38/108 |
| 410 | Après projection | 0.971 | 100/108 | +0.0077 | 0.586 | 108/108 | 34/108 |
| 415 | Courants originaux | 0.935 | 29/108 | -0.0096 | 0.693 | 107/108 | 67/108 |
| 415 | Après projection | 0.948 | 65/108 | +0.0029 | 0.666 | 106/108 | 68/108 |

## Variabilité résiduelle

Le rapport ci-dessous compare les variances temporelles résiduelles/originales dans le bloc test, centrées dans ce bloc uniquement pour calculer la variance. La pente de projection reste apprise dans le premier bloc. Un rapport supérieur à 1 serait possible hors bloc d’ajustement.
- Souris 410 : variance résiduelle médiane = 37.6% de l’originale (36 contributions × 3 graines).
- Souris 415 : variance résiduelle médiane = 65.8% de l’originale (36 contributions × 3 graines).

## Validation et provenance

Tests : retrait exact d’un signal linéaire partagé ; changement massif du bloc test sans modification des coefficients appris ; transfert CCA sous mélange inversible ; conservation des corrélations négatives au test. Identité des cibles, temps, segmentations, régions et provenance entre graines vérifiée. Les empreintes des six PKL sont dans les JSON de résultats. Les scripts et fonctions PCA/corrélation réutilisées sont identifiés dans provenance.json.

Les JSON contiennent chaque comparaison aux six sources, les dix corrélations canoniques, les marges et les fractions de variance conservées par PCA10 dans le premier bloc. Aucun résultat biologique sur l’âge ne découle directement de ce contrôle.

## Interprétation actualisée

Le contrôle original retrouve les nombres historiques : CCA1 89/108 chez 410 et 29/108 chez 415 ; norme 38/108 et 67/108. La nouvelle analyse des dimensions suivantes précise la conclusion : avec la moyenne CCA2–10, la bonne source gagne 108/108 chez 410 et 107/108 chez 415 avant projection. Les marges médianes sont respectivement +0,132 et +0,144. La moyenne des dix dimensions donne les mêmes nombres gagnants. La spécificité ambiguë de CCA1 ne doit donc plus être généralisée à toute la représentation multidimensionnelle.

La projection porte les classements CCA1 à 100/108 et 65/108. Elle ne procure pas un gain général : CCA2–10 reste à 108/108 et passe à 106/108, et la corrélation des normes chez 410 baisse de 0,826 à 0,394. Les sous-espaces demeurent alignables après projection, mais leurs normes ne deviennent pas uniformément plus reproductibles. Aucun de ces résultats ne justifie automatiquement de retirer le signal commun du pipeline.

Pour la maîtrise, conserver la représentation originale et rapporter plusieurs dimensions canoniques, leurs marges de spécificité et les amplitudes séparément. La moyenne CCA2–10 constitue ici un diagnostic exploratoire, pas une nouvelle mesure biologique validée ni un seuil à optimiser. Les axes appris pour chaque paire ne définissent pas encore une caractéristique directement comparable entre sessions spontanées non alignées. La comparaison longitudinale doit encore tenir compte des acquisitions, couvertures et états comportementaux. Les contrôles Narval à 300 passages et tt-1 restent complémentaires.
