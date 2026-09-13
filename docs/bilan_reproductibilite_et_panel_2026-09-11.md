# Bilan et suite scientifique — 11 septembre 2026

## Rectification après lecture des discussions antérieures

Les six graines avaient déjà été rapatriées et analysées le 10 septembre dans « Analyser les discussions de maîtrise », y compris en PCA–CCA et avec des contrôles temporels et régionaux. Le bilan ci-dessous est un **complément sur les traces signées et les unités sans alignement**, pas un premier diagnostic global de reproductibilité. Les dynamiques dominantes sont déjà largement similaires (normes PCA10 : médianes 0,848–0,899 pour 410 et 0,903–0,917 pour 415). La spécificité régionale demeure ambiguë, surtout chez 415 : bonne source première en CCA dans 29/108 comparaisons, contre 89/108 chez 410. Ne pas réduire ce constat aux seules sommes signées. Voir [le contexte consolidé et les bilans sources](contexte_discussions_maitrise.md).

## Résultats acquis

Les six runs Narval 2614685 (410/415 à M6, graines 2026–2028, 100 entraînements) ont été rapatriés et analysés. Les paramètres hors graine, empreintes des données et du code, versions, threads, masques, unités et temps concordent entre graines de chaque souris.

La pVar finale varie de 0,9429 à 0,9464 pour 410 et de 0,9143 à 0,9197 pour 415. Malgré cette proximité, les corrélations temporelles des courants sommés par paire source → cible ont une médiane de 0,359 et 0,623 respectivement. Parmi les 108 comparaisons dépendantes par souris, 35 et 15 sont négatives. Le courant toutes sources présente une corrélation médiane de 0,944 et 0,939. Les diagnostics par unité montrent aussi une variabilité des contributions : l’effet ne se limite pas à la somme des unités cibles.

Ces observations sont compatibles avec des compensations entre sources. Elles ne démontrent ni une décomposition régionale unique ni l’absence de caractéristiques régionales reproductibles. Aucune conclusion sur l’âge n’est permise par ces deux sessions M6.

Rapport, figure et détails : [bilan des graines](../results/reproducibility_2614685/rapport.md). Les six PKL originaux sont conservés ; les cartes unités × temps sont calculées par blocs et peuvent être régénérées à partir des matrices sauvegardées.

## Inventaire longitudinal

Le scan trouve 63 dossiers locaux, dont 57 avec GCaMP, atlas et masque non vides, pour 23 souris. Narval contient 58 dossiers, dont les mêmes 57 complets. Le catalogue du code déclare C2/M8/308, mais le masque requis manque localement et sur Narval. C9/M16/410 existe localement sans atlas ni masque.

Panel exploratoire proposé : 316 à M6/M12/M18, 374 à M6/M10/M18, 415 à M6/M8/M18. Les neuf dossiers ont 5 760 images chacun et des dimensions image/atlas/masque compatibles au sein de chaque dossier. Les dimensions spatiales diffèrent entre sessions. Présence et dimensions ne remplacent pas le contrôle anatomique visuel ni la vérification des métadonnées.

Rapport et inventaires sources : [inventaire longitudinal](../results/longitudinal_inventory/rapport.md). Les étiquettes M sont conservées comme telles ; âge exact, identité longitudinale, dates et multiplicité des sessions restent à confirmer.

## Prochaines étapes

1. Comparer le run 300 de 410/graine2026 au run 100 lorsqu’il sera terminé ; ne pas déduire une stabilité entre graines d’un seul run 300.
2. Exploiter les contrôles temporels et de spécificité déjà réalisés ; examiner les dimensions suivantes, les amplitudes, les cartes par unité et le signal partagé. Étudier la redondance de l’activité, puis la sensibilité à la résolution/segmentation dans des comparaisons appariées.
3. Contrôler visuellement les atlas et masques du panel, confirmer cadence, qualité et métadonnées biologiques ; ne pas utiliser automatiquement les dossiers incomplets.
4. Fixer une procédure commune avant les calculs longitudinaux. Le panel de trois souris suivies tardivement est exploratoire, avec une souris par cohorte ; il ne permet pas d’isoler tous les effets souris/cohorte.

Les scripts ajoutés sont `inventory_longitudinal.py`, `audit_seed_currents.py` et `report_seed_audit.py` dans `scripts/analysis`. Vérifications exécutées : inventaire avec fichier manquant, décomposition matricielle et somme des courants, métriques sous changement d’échelle/décalage/signe et trace constante, compilation Python et inspection visuelle de la figure. Aucun nouvel entraînement Narval n’a été lancé dans ce bilan.
