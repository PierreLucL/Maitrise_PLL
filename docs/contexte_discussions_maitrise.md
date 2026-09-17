# Continuité des discussions de maîtrise

Consolidation du 11 septembre 2026 à la demande de Pierre-Luc. À lire avec [le cap scientifique](cap_scientifique_maitrise.md), qui reste prioritaire. Cette note distingue décisions de l’utilisateur, résultats rapportés et questions ouvertes ; les anciennes réponses ne deviennent pas automatiquement des conclusions validées.

## Sources consultées

- **Questions de maîtrise** — tâche `01a01a4b-8681-7ed2-8a55-68cbc57e8bfc`, échanges du 19 août au 7 septembre.
- **Questions de Narval** — tâche `01a01b4c-ea5c-7021-b5d4-3dfac8cf64b3`, échanges du 19 au 24 août, avec une partie initiale commune.
- **Analyser les discussions de maîtrise** — tâche `01a07936-145f-7760-808c-88aef578c7b9`, échanges du 7 au 11 septembre.

Les outils de lecture des tâches ont été complétés par leurs journaux locaux pour retrouver les messages récents que l’outil renvoyait avec des éléments vides. Les bilans scientifiques correspondants ont été lus directement. Six bilans sont copiés sans modification dans [historique_maitrise](historique_maitrise/provenance.json), avec chemins sources et SHA-256. Cette consolidation n’est pas une nouvelle vérification des articles ni un recalcul de PCA/CCA.

## Décisions à conserver

1. **Objectif biologique :** étudier les interactions fonctionnelles inférées au cours du vieillissement, avec comparaisons longitudinales et entre souris du même âge. La validation technique sert cette étude ; l’unicité des poids et le maximum de pVar ne sont pas les objectifs finaux.
2. **Données déjà en ΔF/F :** Pierre-Luc l’a confirmé le 22 août et a explicitement demandé de retirer `compute_dff`. Ne pas réintroduire un second ΔF/F. « DFF OFF » dans les anciennes expériences signifie absence de recalcul, pas fluorescence brute. Les valeurs négatives et l’échec du double ΔF/F ne prouvent pas que 415 soit corrompue.
3. **Représentation principale :** le 8 septembre, Pierre-Luc a demandé de décortiquer les contributions par unité cible, sans sommer prématurément. Cartes unités × temps, activité du modèle à gauche, sources en colonnes, mêmes unités dans le même ordre sur chaque ligne. Hauteurs proportionnelles aux effectifs, fenêtre temporelle courte configurable, couleurs conservées, barre rectangulaire et esthétique Matplotlib sobre. Une unité widefield est une sous-région de pixels, pas un neurone individuel.
4. **Méthode de comparaison :** chercher à reproduire la méthode publiée des auteurs. L’implémentation locale PCA10–normes–CCA suit une description méthodologique ; l’identité exacte avec les scripts des auteurs n’a pas été établie ligne par ligne. Ne pas la présenter comme acquise.
5. **Préserver la comparaison appariée :** le run 300 de 410/graine2026 doit conserver code, données, segmentation, initialisation et bruit de la référence 100. La correction d’indice `tt → tt-1` relève d’une expérience séparée. Les six runs 100 et ce run 300 utilisent la version historique `tt`.
6. **Choix de travail provisoire :** pixels15, sigma4, g0,8, ampInWN0,01, GSR désactivée ; ces réglages ne sont pas une validation universelle. Ne pas remettre en route un grand screening ou modifier le cœur pour gagner du temps sans une question scientifique et un contrôle apparié. Les essais CPU historiques n’ont pas montré une accélération proportionnelle au nombre de cœurs.
7. **Organisation :** un seul pipeline configurable, chemins portables via argument puis `MAITRISE_DATA_DIR` puis `data/`, sorties dans `results/`. Transfert Narval limité aux fichiers du loader. Préserver résultats, logs et snapshots ; une ancienne synchronisation avec suppression avait effacé un log. Les commentaires de code doivent rester pédagogiques, avec le ton québécois simple demandé.

## Ce qui était déjà acquis avant cette tâche

### Les six graines avaient déjà été analysées le 10 septembre

Les analyses de 410 puis de 415 se trouvaient dans l’autre dossier de travail. Il ne fallait donc pas présenter le rapatriement et le bilan des six graines comme entièrement nouveaux. Notre audit du 11 septembre est un complément sur les traces signées et les corrélations par unité.

| Souris | Médianes de corrélation des normes PCA10, selon les 3 paires de graines | Médianes CCA1 | Moyenne des 10 CCA : médianes selon les paires |
|---|---|---|---|
| 410 | 0,899 ; 0,855 ; 0,848 | 0,987 ; 0,987 ; 0,986 | 0,715 ; 0,706 ; 0,715 |
| 415 | 0,903 ; 0,908 ; 0,917 | 0,988 ; 0,989 ; 0,989 | 0,755 ; 0,755 ; 0,774 |

Sources : [410](historique_maitrise/graines_410.md), [415](historique_maitrise/graines_415.md). Ces résultats sont descriptifs et calculés sur des matrices unités × temps centrées temporellement, avec dix composantes non blanchies. La première CCA décrit le meilleur alignement et ne garantit ni les signes, ni les amplitudes, ni toutes les dimensions. Les distributions des blocs de J sont proches malgré une faible concordance des poids individuels ; cela ne constitue pas une preuve biologique.

### Les contrôles temporels et régionaux ont déjà été faits

PCA et CCA ajustées sur 0–230 s, puis projection et évaluation sur 250 s–fin avec axes figés. Quarante décalages circulaires servent de contrôles descriptifs.

| Souris | CCA test médiane | Bonne source meilleure que les 5 autres en CCA | Marge médiane sur la meilleure autre source | Bonne source meilleure selon la norme |
|---|---|---|---|---|
| 410 | 0,982 | 89/108 | +0,0022 | 38/108 |
| 415 | 0,935 | 29/108 | −0,0096 | 67/108 |

Les 108 comparaisons par souris dépassent le P95 des décalages testés, pour la norme et la CCA. Elles sont dépendantes ; ni ce percentile ni le classement parmi six sources ne sont des p-values. Le réseau a appris toute la session : on teste le transfert temporel de l’alignement, pas la généralisation du réseau à une session inédite. Source : [contrôles](historique_maitrise/controles_specificite.md).

**Interprétation commune : les dynamiques dominantes sont reproductibles, les amplitudes et les signes le sont moins, et la spécificité de source reste ambiguë, surtout chez 415.** Une dynamique commune pourrait contribuer aux scores élevés ; son origine reste à déterminer.

### La version et les anciennes comparaisons demandent de la prudence

- Le test synthétique apparié `tt`/`tt-1` existe déjà : effet variable, sans amélioration uniforme. Le pilote apparié sur données réelles reste ouvert. Voir [test synthétique](historique_maitrise/correction_indice_synthetique.md).
- Les anciennes comparaisons 100/300/1000 sans graines et bruit sauvegardés n’isolent pas l’effet du nombre de passages.
- `J_final` seul ne permet pas de reconstituer la trajectoire historique exacte. Utiliser les trajectoires et entrées sauvegardées ; ne pas confondre une resimulation approximative et la reconstruction évaluée pendant le run.
- `nRunFree` réutilise état/bruit/matrice : sa stabilité n’est pas une répétition indépendante.
- Le score historique concaténé dépendait des amplitudes et de l’affichage. Le centrage et la division par √N sont des choix d’affichage, pas une calibration biologique. Aucun recadrage temporel silencieux ; caches liés à la provenance.

Source : [audit du 7 septembre](historique_maitrise/audit_courants_7_septembre.md).

## Réconciliation avec le bilan du 11 septembre

Les médianes signées 0,359 (410) et 0,623 (415) calculées ici ne contredisent pas les résultats PCA–CCA : les objets comparés sont différents. Elles ne permettent pas à elles seules de déclarer les contributions régionales globalement non reproductibles. Les corrélations par unité sans alignement fournissent également un diagnostic distinct d’un sous-espace PCA–CCA. L’audit reste utile pour les signes et amplitudes, mais doit être présenté comme complémentaire.

Les cinq dossiers C0/M20/42, C2/M6/308, C5/M12/353, C5/M14/353 et C9/M16/410 avaient déjà été exclus sur demande de Pierre-Luc en août. Leur découverte dans l’inventaire local n’est pas nouvelle et ne les rend pas éligibles au panel. Le masque manquant de C2/M8/308 est une divergence supplémentaire observée le 11 septembre : le catalogue et les anciens bilans annonçaient 58 dossiers complets, le scan actuel en trouve 57. Ne pas inventer la cause de cet écart.

Le nouvel inventaire longitudinal et le panel exploratoire restent utiles. Les trois souris à suivis tardifs appartiennent à trois cohortes différentes ; présence de fichiers et concordance dimensionnelle ne valent pas validation anatomique ou biologique.

## Suite actualisée

1. Exploiter les contrôles de spécificité existants : examiner les dimensions suivantes, les normes/amplitudes, le signal partagé et les cartes par unité, particulièrement chez 415. Ne pas repartir de zéro sur les mêmes contrôles.
2. Caractériser la redondance et le spectre de l’activité ; si nécessaire, comparer des résolutions avec mêmes graines pour rechercher un compromis reconstruction/spécificité/reproductibilité. Le rapport nominal T/N ≈ 2,27 chez 410 ne compte pas les contraintes indépendantes ; interpoler ou entraîner plus longtemps n’ajoute pas d’observations expérimentales.
3. La comparaison 100/300 est terminée le 13 septembre (voir ci-dessous). Décider ensuite d’une éventuelle réplication à 300 pour les autres graines ; tester la correction d’indice séparément avant de choisir la référence de futurs entraînements.
4. Poursuivre le contrôle anatomique et les métadonnées du panel longitudinal. Sélectionner les caractéristiques fiables et comparables avant l’analyse de l’âge ; conserver la souris comme unité biologique.

Cette note consolide le contexte et corrige les priorités ; elle ne déclenche pas de nouvelle soumission ou de modification du moteur CURBD.

## Mise à jour du 13 septembre : run 300 rapatrié et comparé

Le job 2796272_0 a terminé avec succès en 56 h 06 min 51 s. Le PKL local est identique à l’empreinte SHA-256 calculée sur Narval. Les modèles 410/graine2026 à 100 et 300 passages ont mêmes données, temps, masque, unités, J0, bruit, état initial, iTarget, facteur de normalisation, paramètres hors durée et empreintes de code. Les 100 premières pVar enregistrées sont exactement identiques.

- pVar recalculée : 0,945365 → 0,958582 ; amélioration des six régions, y compris de la corrélation des dérivées regroupées par région.
- Normes PCA10 : corrélation médiane 0,942 ; CCA1 médiane 0,992 ; médiane des moyennes des dix CCA 0,936.
- Ratio moyen des normes PCA10 300/100 : médiane 1,248, étendue 1,029–1,862. Les dynamiques restent proches, mais l’amplitude de cette représentation change avec la durée.
- Alignement PCA/CCA appris avant 230 s et testé après 250 s : CCA1 test médiane 0,990 ; bonne source première dans 36/36 cas, marge médiane seulement +0,0048. Ce test est entre durées pour une même graine, pas entre graines à 300 ; il ne démontre pas une amélioration de la spécificité inter-graines.

Les cartes 60–120 s utilisent les mêmes unités, le même ordre fixé depuis le modèle 100 et une échelle partagée. Les métriques signées restent complémentaires. Un seul modèle à 300 ne permet pas de conclure à sa reproductibilité entre graines ou à la stabilité définitive des amplitudes. Aucun nouveau modèle n’a été entraîné dans cette comparaison.

[Bilan détaillé et méthode](../results/comparison_train100_300_410_2026/bilan.md) ; scripts `scripts/analysis/compare_training_durations.py` et `report_training_durations.py`. Les contrôles synthétiques PCA/SVD, CCA sous mélange inversible, centrage sur le seul bloc d’ajustement et trace constante passent ; les quatre figures ont été inspectées visuellement.

## Mise à jour du 13 septembre : diagnostic partagé et contrôle du panel

Sur les six modèles à 100 passages, projection descriptive de chaque courant par unité sur la moyenne des six activités régionales (toute la session, sans retrait du signal dans le pipeline). Médiane de la fraction de variance associée sur 36 blocs × 3 graines : 69,5 % chez 410, 50,8 % chez 415. Le signal partagé est donc marqué, mais son niveau ne suffit pas à expliquer la moins bonne spécificité précédemment observée chez 415. Ce diagnostic ne prouve ni causalité, ni artefact global, ni effet du sigma ; il n’évalue pas les résidus en PCA/CCA hors bloc d’ajustement.

Neuf sessions du panel contrôlées par planche atlas nettoyé/masque/variabilité de 32 frames : six régions présentes partout, 98,54–100 % du masque attribué, valeurs échantillonnées finies. Couvertures régionales variables ; aucune validation de registration ou d’alignement anatomique entre sessions. Les descriptions TIFF donnent les dimensions uniquement. Aperçu partiel de Datasets.numbers inspecté : catalogue souris × mois par cohorte ; classeur complet non lu, accès UI bloqué par permissions. Métadonnées biologiques/acquisition toujours non confirmées.

[Bilan, méthode et suites](../results/diagnostic_partage_panel_2026-09-13/bilan.md). Aucun nouvel entraînement ou changement du moteur.

## Soumission du 13 septembre : contrôles lourds pendant les analyses locales

À la demande de Pierre-Luc, job Narval **2989725**, array 0–6 limitée à deux tâches simultanées : deux pilotes `tt-1` à 100 passages (410/415, graine2026), puis cinq compléments historiques `tt` à 300 passages (410 graines2027/2028 ; 415 graines2026/2027/2028). Le 410/2026/300 existant n’est pas relancé. Sigma4 et tous les autres réglages restent fixes. Le moteur du dépôt courant est inchangé.

Snapshot `paired_controls_20260913_v1` construit depuis `repro15_b3abdc905a6b5302`, deux copies ne différant que par la ligne de r_slice. Tests et prévols réussis ; 4 CPU et 16 Go par tâche, 36 h de limite pour les pilotes100, 96 h pour les runs300. État au contrôle de soumission : sept tâches en attente, démarrage non déterminé. Ce sont des calculs soumis, pas des résultats. Vérifications automatiques de provenance/versions/threads, appariement de la cible avant entraînement et des entrées/initialisation après ; préfixe pVar100 exact exigé pour le volet durée.

[Protocole et reçu](../results/narval_paired_controls_20260913/protocole.md). Le volet indice reste un pilote à une graine par souris ; il ne valide pas à lui seul la reproductibilité de la correction. Les contrôles locaux du signal partagé peuvent continuer pendant l’attente.

## Mise à jour du 14 septembre : spécificité multidimensionnelle et signal partagé

Contrôle local terminé sur les six modèles 100. Projection par unité sur la moyenne des six activités régionales, coefficients appris avant 230 s ; PCA 10 et CCA apprises sur ce bloc, axes/signes figés et évaluation après 250 s. Aucun réentraînement ou changement de GSR. Le contrôle original retrouve exactement les classements historiques CCA 1 et norme.

Résultat à conserver : **l’ambiguïté de source de CCA 1 ne s’étend pas à toutes les dimensions**. La moyenne des corrélations test CCA 2–10 classe la bonne source première dans 108/108 cas chez 410 et 107/108 chez 415, avant projection ; marges médianes +0,132/+0,144. La moyenne des dix CCA donne les mêmes nombres. Après projection : CCA 1 gagne 100/108 et 65/108 (contre 89/108 et 29/108) ; CCA 2–10 gagne 108/108 et 106/108. Les corrélations test CCA 1 restent élevées (0,971/0,948), mais la norme chez 410 devient moins reproductible (médiane 0,826→0,394).

Interprétation : information régionale reproductible dans plusieurs dimensions, premier alignement partiellement ambigu ; pas de bénéfice uniforme de la projection. Ne plus présenter 415 comme globalement dépourvue de spécificité sur la seule base de CCA 1. Ne pas confondre ce classement descriptif dépendant avec causalité, validation biologique ou généralisation du RNN à de nouvelles sessions. Les axes CCA de chaque paire ne définissent pas encore une mesure intersessions directement comparable. Garder amplitudes, normalisation et contrôle anatomique comme questions distinctes.

[Bilan et figure](../results/shared_signal_specificity_20260914/bilan.md). Trois tests dédiés passent (projection et absence de fuite, mélange inversible, signe test conservé) ; figure inspectée.

## Rapatriement du 15 septembre : pilotes tt-1

Les tâches2989725_0 (410) et2989725_1 (415), graine2026,100 passages, ont terminé avec succès en23h52 et20h27. pVar finales rapportées :0,962845 et0,927272 ; références historiques0,945365 et0,914296. Appariement des entrées et initialisations confirmé sur Narval. Les deux PKL, CSV et preuves d’appariement ont été rapatriés : six empreintesSHA-256 identiques aux sources distantes, environ1,025Go. Journaux également copiés. Voir `results/narval_paired_controls_20260913/2989725/transfer_verified.json`. La comparaison détaillée des courants et traces entre versions reste à effectuer. Au contrôle du15septembre21h28UTC, les tâches2/3 (410,300 passages) tournaient depuis27h24/24h02 ; les tâches4–6 attendaient.

## Comparaison des indices terminée le 15 septembre

Pilotes appariés tt/tt-1, 100 passages, graine2026, souris410/415 : entrées et initialisations identiques, sources vérifiées contre snapshots ne différant que par r_slice. pVar recalculée410 :0,945365→0,962845 (erreur quadratique −32,0%) ;415 :0,914296→0,927272 (−15,1%). Amélioration dans les six régions de chaque souris.

Courants par unité conservés : corrélation médiane des normes PCA10 entre versions0,989/0,994 ; médiane des moyennes CCA10 sur session0,985/0,992 ; médiane des corrélations par unité puis par bloc0,986/0,990. CCA2–10 test (axes appris avant230s, évalués après250s)0,982/0,989 ; source correspondante première36/36 chez chacune. Ce résultat entre versions à même graine n’est pas une validation inter-graines de tt-1.

Amplitude RMS centrée tt-1/tt : médiane0,849 chez410 (étendue0,703–1,251),0,912 chez415 (0,818–1,029). Les dynamiques sont très proches, les amplitudes ne sont pas interchangeables. Pilotes favorables à tt-1, mais ne pas mélanger des versions dans l’étude d’âge ; reproductibilité du nouvel indice encore à établir. Le moteur de production reste inchangé.

[Bilan et provenance](../results/comparison_indices_20260915/bilan.md). Trois figures inspectées ; traces de six unités par souris choisies près de la pVar régionale médiane du modèle tt, mêmes échelles, aucun lissage ajouté. Scripts compare_index_variants.py, report_index_variants.py et compare_reconstruction_indices.py.

## Réplications tt-1 soumises le 16 septembre

À la demande de Pierre-Luc, quatre réplications à100 passages :410/2027,410/2028,415/2027,415/2028. Job array3178099 (0–3%2),4CPU/16Go/36h par tâche. Snapshot ttminus1_replicates_20260916_v1 reprend exactement le moteur corrigé des pilotes2026 ; paramètres conservés. Tests et quatre prévols réussis ; reçu local. Le contrôle des cibles prétraitées se fait au démarrage effectif, puis les initialisations/entrées sont vérifiées en fin de run. Les calculs historiques300 ne sont pas modifiés. Après rapatriement, comparer trois graines par souris pour reconstruction, dynamique multidimensionnelle, spécificité et amplitudes ; aucune validation inter-graines de tt-1 acquise avant ces résultats.

[Protocole et reçu](../results/narval_ttminus1_replicates_20260916/protocole.md).

## Audit des masques du 16 septembre

Lecture de la chaîne active et contrôle des masques sauvegardés des pilotes410/415 : aucun décalage d'IDs détecté entre masque, métadonnées, regions et dimensions des traces. Attention aux conventions : clean_reduced_atlas n'est pas appelée par prepare_timeseries ; NaN/non-mappés restent exclus, tandis que le fond0 peut être réétiqueté par le nettoyage initial. pixels15 est une taille cible KMeans, sans garantie de connexité :3/2538 et8/2331 parcelles non connexes à4voisins. Noms anatomiques à clarifier : selon la table commentée, M.II inclut AUD*, Ass. contient VISa/VISrl, Vis. inclut TEa ; légende originale atlas.npy à vérifier avant interprétation anatomique. Les IDs spatiaux ne sont pas homologues entre sessions. Aucun code de segmentation ni snapshot Narval modifié. Voir results/audit_masks_20260916/bilan.md et les contrôles synthétiques reproductibles.

## Intention et prototype de segmentation clarifiés le16septembre

Pierre-Luc veut corriger les artefacts du masque anatomique initial, notamment des lignes fines étiquetées comme une région éloignée, avant la parcellisation. Préserver aveuglément le parent initial serait donc trop restrictif. Nouvelle voie locale explicite coherent_v1 : nettoyage historique conservé, correction complémentaire prudente par soutien local/distance à des noyaux, séparation des parcelles non connexes, fusion des singletons adjacents dans le parent corrigé, contrôle pixel→parent→ID→trace→regions. Voie historique par défaut et snapshots Narval conservés.33tests passent. Sur410 :22pixels réattribués supplémentaires,2538unités,0singleton et0parcelle non connexe ;415 :30pixels,2335unités,3singletons isolés signalés,0parcelle non connexe. Cas anatomiques ambigus non forcés. Proposition testée sur deux masques, non validée longitudinalement ; voir docs/segmentation_coherente.md. La nomenclature des six groupes demeure à confirmer.
