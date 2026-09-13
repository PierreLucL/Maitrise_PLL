# Reproductibilité des courants — job 2614685

## Conclusion

Cet audit complémentaire porte sur les sommes signées et les corrélations par unité sans alignement. Certaines traces sont anticorrélées entre graines, mais cela ne contredit pas la reproductibilité des dynamiques dominantes déjà observée en PCA–CCA. Il faut distinguer dynamique, signe, amplitude et spécificité régionale avant de choisir une signature du vieillissement.

## Mise en contexte historique — ajout du 11 septembre

Les six modèles avaient déjà été rapatriés et analysés le 10 septembre dans « Analyser les discussions de maîtrise ». Les corrélations médianes des normes PCA10, selon les trois paires de graines, vont de 0,848 à 0,899 pour 410 et de 0,903 à 0,917 pour 415 ; les premières CCA médianes sont respectivement 0,986–0,987 et 0,988–0,989. Le calcul suit une description méthodologique, sans identité vérifiée ligne par ligne avec les scripts des auteurs.

Les contrôles temporels et de source existent également : CCA test médiane 0,982 / 0,935 ; bonne source première en CCA 89/108 / 29/108, pour 410 / 415. Ces comparaisons sont dépendantes et le réseau a appris la session entière. Le point ouvert est donc notamment la spécificité régionale, surtout chez 415, et non une absence générale de reproductibilité. Voir ../../docs/contexte_discussions_maitrise.md et les bilans sources qui y sont référencés.

## Données et méthode

Deux dossiers C9/M6 (410 et 415), trois graines (2026, 2027, 2028), 100 entraînements, segmentation fixée à 0. Les six sorties sont locales. Vérifications réussies entre graines de chaque souris : paramètres hors graine, empreinte des données, code, versions, threads, masque (NaN hors cortex), identités des unités et grille temporelle.

Courant calculé comme J[cible, source] × RNN[source, temps], avec les tableaux float32 sauvegardés. Comparaison temporelle de toute la session entre graines de la même session. Analyse des sommes régionales et des traces de chaque unité cible, sans aligner les souris entre elles. Les cartes unités × temps sont calculées par blocs ; les PKL originaux conservent leurs facteurs pour les régénérer sans perte supplémentaire.

## Résultats descriptifs

| Souris | pVar finale (min–max) | Médiane r des sommes source→cible | Comparaisons r < 0 | Médiane r du courant toutes sources |
|---|---|---|---|---|
| 410 | 0.9429–0.9464 | 0.359 | 35/108 | 0.944 |
| 415 | 0.9143–0.9197 | 0.623 | 15/108 | 0.939 |

Chaque souris fournit 36 paires anatomiques × 3 comparaisons de graines = 108 valeurs ; ces valeurs sont dépendantes et ne sont pas 108 répétitions biologiques. Le courant toutes sources donne 6 cibles × 3 comparaisons = 18 valeurs. Les médianes ne constituent pas un test statistique.

### Souris 410

- Corrélation des sommes source→cible : minimum -0.869, maximum 0.948.
- Rapport d’écart-type b/a : 0.22 à 3.32. Une corrélation élevée ne garantit donc pas une amplitude comparable.
- Médiane des médianes par unité, par paire anatomique et paire de graines : 0.465.
- Courant récurrent toutes sources : r régional entre 0.923 et 0.963 ; médiane des médianes par unité 0.936.
- Activité RNN : médiane des corrélations médianes par unité et cible 0.963.

### Souris 415

- Corrélation des sommes source→cible : minimum -0.843, maximum 0.938.
- Rapport d’écart-type b/a : 0.25 à 2.61. Une corrélation élevée ne garantit donc pas une amplitude comparable.
- Médiane des médianes par unité, par paire anatomique et paire de graines : 0.547.
- Courant récurrent toutes sources : r régional entre 0.908 et 0.975 ; médiane des médianes par unité 0.829.
- Activité RNN : médiane des corrélations médianes par unité et cible 0.926.

## Conséquence pour le programme longitudinal

La stabilité du courant total malgré des contributions variables serait compatible avec des compensations entre sources ; elle ne démontrerait pas une décomposition régionale unique. La figure montre également les corrélations par unité pour éviter de conclure uniquement à partir de sommes régionales.

1. À la fin du run 300 de 410/graine2026, vérifier à données et segmentation identiques si les contributions et leurs amplitudes changent par rapport au run 100. Un seul run 300 ne teste pas la reproductibilité à 300.
2. Si cette prolongation est utile, compléter de façon ciblée les autres graines à 300 pour 410 avant de généraliser. Choisir selon la stabilité des caractéristiques et le coût, pas seulement pVar.
3. Pendant ce temps, préparer le panel longitudinal décrit dans ../longitudinal_inventory/rapport.md : trois souris avec dossiers M6 et M18, plus âges intermédiaires. Confirmer les métadonnées et faire le contrôle qualité anatomique.
4. Exploiter les contrôles temporels et régionaux déjà réalisés : examiner les dimensions suivantes, le signal partagé, les amplitudes et les cartes par unité, particulièrement chez 415. Étudier la redondance de l’activité et la sensibilité à la normalisation avant de choisir les caractéristiques biologiques.

## Limites

- Deux souris à M6 seulement : aucune conclusion sur le vieillissement.
- Reconstruction sur la session entraînée ; aucune validation hors échantillon ici.
- Segmentation fixée : sa variabilité reste à étudier.
- Les courants sont des interactions fonctionnelles inférées ; ni connexions synaptiques ni preuve causale.
- Métadonnées de session issues des sorties et dossiers ; les empreintes établissent la concordance entre graines, pas l’exactitude biologique des étiquettes.

## Reproduction

`OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 MPLCONFIGDIR=/tmp/curbd-mpl python3 scripts/analysis/audit_seed_currents.py results/narval_reproducibility_pix15_410_415/2614685 results/reproducibility_2614685`

`MPLCONFIGDIR=/tmp/curbd-mpl python3 scripts/analysis/report_seed_audit.py`

Les détails numériques, métriques par unité, sources et vérifications sont dans mouse410.json et mouse415.json.
