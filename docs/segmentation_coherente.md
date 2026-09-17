# Masques cohérents pour la comparaison longitudinale

Intention précisée par Pierre-Luc : éliminer les artefacts d'étiquettes anatomiques (ex. ligne d'un pixel attribuée à une région éloignée), puis créer des petites parcelles spatiales cohérentes, minimiser les singletons et garantir leur rattachement au bon parent dans CURBD.

## Implémentation candidate explicite

Dans la configuration, `segmentation_method: coherent_v1` active la nouvelle voie. Sans cette option, la segmentation historique reste utilisée. Les snapshots Narval ne changent pas. Le nom du résultat inclut l'option lorsqu'elle est spécifiée, pour distinguer les méthodes.

1. Nettoyage historique sur l'atlas détaillé : fenêtre5×5 et seuil0,25 ; réduction en six parents et application du ROI.
2. Correction complémentaire conservatrice sur les parents : noyaux obtenus par érosion3×3 ; pixels dont le parent représente moins de25% du voisinage5×5 ; réattribution seulement si le noyau gagnant est unique, à3pixels au plus et plus proche de plus de1pixel que celui du parent initial. Fond non rempli. Pixels ambigus signalés ; aucun noyau inventé. Il s'agit d'une heuristique à valider visuellement, pas d'une vérité anatomique. Les distances sont en pixels, leur comparabilité exige une échelle spatiale comparable entre sessions.
3. KMeans spatial historique, graine de segmentation fixe. Séparation des composantes4-connexes de chaque parcelle ; fusion des singletons vers une parcelle adjacente du même parent corrigé, priorité à la plus grande puis au plus petit ID à égalité. Aucun saut à travers le fond. Les singletons isolés sont conservés et signalés.
4. IDs contigus triés spatialement, métadonnées reconstruites et tableau regions recalculé depuis ces métadonnées.
5. Contrôle automatique avant CURBD : mêmes pixels couverts que le masque parent, parent unique et exact par parcelle, nombre de pixels exact, IDs0..N−1, nombre de traces identique, chaque ID une fois dans regions et bon nom de parent. Ce contrôle protège aussi la voie historique sans modifier ses pixels.

Les résultats enregistrent `parent_mask` et `segmentation_qc` (méthode, paramètres de correction, transitions de parents, pixels ambigus, singletons et connexité). Les masques bruts restent dans les fichiers d'entrée. Les noms des six groupes restent ceux de la table existante ; cet ajout ne valide pas leur nomenclature anatomique.

## Validation

Tests synthétiques : ligne aberrante loin de son propre noyau, fond conservé, absence de noyau, parent erroné, IDs manquants/dupliqués/non contigus, connexité, fusion sans changement de parent, moyennes de pixels connues et appel de bout en bout à prepare_timeseries. Suite complète :33tests réussis le16septembre2026. Un ancien fixture de test a été rendu cohérent avec les parents qu'il simulait ; aucune assertion scientifique supprimée.

Aperçus réels dans results/audit_masks_20260916/coherent_410.* et coherent_415.*. Avant adoption pour le panel longitudinal : inspecter les changements, les couvertures et les échelles spatiales sur plusieurs sessions/âges. Garder une seule version pour les comparaisons biologiques. Une validation sur deux masques n'établit pas la robustesse longitudinale.
