# Contrôles appariés soumis sur Narval — 13 septembre 2026

Objectif : déterminer si la durée d’apprentissage rend les caractéristiques des courants plus reproductibles et spécifiques, puis mesurer séparément l’effet de l’indice temporel avant de choisir la version pour le pilote longitudinal.

| Tâche Slurm | Version | Souris | Graine | Passages |
|---|---|---|---|---|
| 0 | tt-1, pilote | 410 | 2026 | 100 |
| 1 | tt-1, pilote | 415 | 2026 | 100 |
| 2 | tt historique | 410 | 2027 | 300 |
| 3 | tt historique | 410 | 2028 | 300 |
| 4 | tt historique | 415 | 2026 | 300 |
| 5 | tt historique | 415 | 2027 | 300 |
| 6 | tt historique | 415 | 2028 | 300 |

Le modèle 410/2026/300 terminé n’est pas relancé. Les six références à 100 sont conservées. Le volet durée donnera trois graines à 100 et 300 par souris ; les graines restent des répétitions techniques. Le volet indice ne comporte qu’une graine par souris et ne constitue pas une validation inter-graines de la version corrigée.

Paramètres conservés : pixels15, sigma4, dtData=1/12, dtFactor2, tauRNN0.33, g0.8, ampInWN0.01, GSR désactivée, segmentation_seed0, 20 passages libres. Aucun second ΔF/F. La version corrigée ne diffère du code historique que par `r_slice = RNN[iTarget, tt-1]` à la place de `tt`.

## Exécution et provenance

Code historique rapatrié depuis `repro15_b3abdc905a6b5302`, empreintes vérifiées contre son manifeste et celles inscrites dans les six PKL de référence. Deux copies isolées dans `snapshot/legacy` et `snapshot/corrected`. Le cœur du projet courant n’est pas modifié.

Snapshot distant : `/scratch/pllar11/Maitrise_PLL/run_snapshots/paired_controls_20260913_v1`.

Une array de sept tâches, maximum deux simultanées ; quatre CPU, 16 Go, limite 96 heures pour les cinq runs 300 et 36 heures pour les deux pilotes 100 (limites des tâches 0 et 1 abaissées avec scontrol après soumission). Le précédent 410/300 a pris 56 h : ordre de grandeur d’une semaine pour ce lot, hors attente, avec incertitude sur les autres graines et 415. La limite Slurm n’est pas une estimation de durée réelle.

Avant soumission : test synthétique apparié local ; suite de reproductibilité pour chaque version sur Narval ; vérification des versions des paquets, des threads, des empreintes de code et PKL de référence, des fichiers sources et dry-run des sept configurations. Le manifeste inclut les scripts d’exécution et configurations.

Avant chaque entraînement : vérification de l’empreinte des traces prétraitées, de la segmentation et des régions contre la référence. Après chaque entraînement : identité de Adata, temps, J0, bruit, état initial, iTarget, normalisation et paramètres hors durée. Pour le volet durée, les 100 premières pVar doivent être strictement identiques à la référence. Un échec de vérification remonte à Slurm. `pairing_verified.json` est écrit seulement en cas de succès.

Soumission protégée par verrou et reçu distant : ne pas resoumettre une tentative ambiguë sans vérifier Slurm. Aucun effacement de données ou résultats.

## Analyse au retour

Comparer reconstruction et dérivées, cartes unités × temps, normes PCA10, dimensions CCA, amplitudes et corrélations par unité. Reprendre le contrôle de spécificité de source avec axes appris avant 230 s et évalués après 250 s. Garder séparés effet de durée et effet de version. Le contrôle local du signal commun peut avancer pendant les entraînements.

Une hausse de pVar ou de CCA1 seule ne justifie pas de choisir 300 ou d’adopter la correction pour tout le panel. La décision dépend de la reproductibilité, de la spécificité et de la comparabilité des caractéristiques longitudinales.

## Soumission confirmée

Job **2989725**, sept tâches acceptées. Les deux suites de reproductibilité et les sept prévols ont réussi. Au dernier contrôle, toutes les tâches sont en attente (Priority ou None), sans date de démarrage disponible. L’estimation affichée par `sbatch --test-only` concernait une simulation avec toutes les limites à 96 h ; elle ne constitue pas une réservation ni une date de départ confirmée du job réel. Reçu et logs de prévol rapatriés dans `receipt/`.
