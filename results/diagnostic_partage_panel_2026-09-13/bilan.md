# Signal partagé et panel longitudinal — 13 septembre 2026

## Question scientifique

Déterminer quelles caractéristiques des courants peuvent être comparées entre sessions et âges, en séparant composante partagée, spécificité régionale et couverture anatomique. Diagnostic descriptif des six modèles existants à 100 passages ; aucun nouvel entraînement, retrait du signal global ou changement de sigma.

## Signal partagé : méthode

Le prédicteur g(t) est la moyenne non standardisée des six moyennes régionales de Adata (activité déjà prétraitée et normalisée). Chaque région a le même poids. Les courants sauvegardés sont calculés par J[cible,source] @ RNN[source], en float64, aux temps RNN les plus proches des 5 760 temps des données. Chaque unité cible est centrée temporellement puis projetée sur g centré. La fraction rapportée est la somme des énergies projetées divisée par la somme des énergies centrées des unités. Ce n’est pas une somme signée des courants.

La projection est ajustée et évaluée sur toute la session : aucune interprétation comme performance hors échantillon ou mesure causale. Le prédicteur inclut la région source et la cible et provient des données apprises par le réseau. Une fraction élevée ne prouve ni artefact ni origine exclusivement globale. Les corrélations des activités sont une seule observation par souris ; les graines ne sont pas des répétitions biologiques.

| Souris | Fraction activité associée à g | Corrélation régionale médiane hors diagonale | Fraction courants médiane, 36 blocs × 3 graines | Médianes par graine 2026/2027/2028 |
|---|---:|---:|---:|---|
| 410 | 81.0% | 0.949 | 69.5% | 75.5% / 73.5% / 60.5% |
| 415 | 65.4% | 0.808 | 50.8% | 53.5% / 46.3% / 51.7% |

Les figures montrent les moyennes par bloc sur trois graines ; les médianes du tableau portent sur les 108 blocs, ce qui explique une différence de résumé. Le signal partagé est marqué chez les deux souris et davantage chez 410 selon cette définition. Il ne suffit donc pas à expliquer à lui seul la moins bonne spécificité inter-graines précédemment observée chez 415. Ce résultat motive une analyse des dimensions résiduelles, sans décider automatiquement de retirer le signal global.

## Panel : contrôle effectué

Neuf sessions : 316 M6/M12/M18, 374 M6/M10/M18, 415 M6/M8/M18. Atlas nettoyé avec les paramètres de production (size=5, min_fraction=0.25), puis regroupement en six régions avec le masque ROI. Fond de figure : écart-type de 32 frames réparties sur la session, contraste P2–P98 propre à chaque image. Ce fond n’est pas une image anatomique et ne permet pas de valider une registration.

| Souris | Dossier | Pixels du masque attribués aux six régions | Six régions présentes |
|---|---|---:|---|
| 316 | M6 | 100.00% | oui |
| 316 | M12 | 99.98% | oui |
| 316 | M18 | 100.00% | oui |
| 374 | M6 | 99.57% | oui |
| 374 | M10 | 99.74% | oui |
| 374 | M18 | 99.79% | oui |
| 415 | M6 | 98.54% | oui |
| 415 | M8 | 99.49% | oui |
| 415 | M18 | 98.95% | oui |

Toutes les valeurs des 32 frames échantillonnées à l’intérieur des masques sont finies ; le film complet n’a pas été contrôlé pour ce diagnostic. Les six régions sont présentes, avec 98,54 à 100 % du masque attribué. Les couvertures et la variabilité spatiale changent entre sessions : par exemple la région Vis chez 374 occupe 3 288 pixels à M6 et 2 278 à M18. Cela peut refléter cadrage, masque ou atlas ; ce n’est pas une variation biologique de surface démontrée. Les dimensions spatiales et tailles des régions ne justifient pas une correspondance directe des sous-unités entre sessions.

## Métadonnées et limites restantes

La recherche des documents usuels sous /Volumes/Toute ma vie/Datasets trouve Datasets.numbers et outline_regions 1.txt. Les descriptions de première page des neuf TIFF ne contiennent que les dimensions. L’aperçu embarqué de Datasets.numbers a été inspecté : il montre des tableaux cohorte, mois, souris. Il est partiel ; le classeur complet n’a pas été lu. L’accès Numbers via contrôle d’écran est bloqué par les permissions. Âges exacts, dates, sexe, cadence et comportement restent non confirmés. Pierre-Luc indique penser que les métadonnées sont dans Datasets, sans en connaître la localisation exacte.

## Prochaine décision méthodologique

1. Quantifier la reproductibilité et le classement des sources après projection diagnostique du signal commun, avec apprentissage de la projection et de PCA/CCA sur le premier bloc et évaluation sur le second. Comparer aussi les dimensions CCA suivantes. Il s’agit d’un contrôle de sensibilité sur modèles existants, pas d’une modification du prétraitement.
2. Examiner les variations de couverture par région et obtenir des références anatomiques et métadonnées avant de valider définitivement le pilote M6→M18.
3. Garder sigma et durée comme choix provisoires. Ce diagnostic n’isole pas l’effet du lissage ; il ne justifie pas à lui seul un nouveau screening.

## Vérification et fichiers

La formule de projection a été vérifiée sur signal entièrement partagé, signal orthogonal et mélange de variance connue. Les deux figures ont été inspectées. Les JSON conservent chemins, paramètres descriptifs et indices des frames. provenance.json contient les empreintes des six modèles et du script.

- [Signal partagé](signal_partage.png)
- [Atlas du panel](panel_atlas.png)
- [Mesures des courants](shared_signal.json)
- [Contrôle du panel](panel_qc.json)
