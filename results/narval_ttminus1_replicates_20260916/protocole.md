# Réplications tt-1 — étape 1, 16 septembre 2026

Autorisation : Pierre-Luc demande de répliquer tt-1 avec les autres graines.
Objectif : vérifier la reproductibilité des caractéristiques des courants avant leur utilisation dans l'étude longitudinale du vieillissement.

Quatre tâches : 410/2027, 410/2028, 415/2027, 415/2028. Toutes à100 passages +20 libres, variante corrigée identique aux pilotes2026. Pixels15, segmentation_seed0, sigma4, tau0,33, dtFactor2, g0,8, ampInWN0,01, P0=1, GSR désactivée. Les pilotes2026 ne sont pas relancés.

Snapshot ttminus1_replicates_20260916_v1 construit à partir du manifeste vérifié paired_controls_20260913_v1. Sources du moteur inchangées ; seuls les configurations, scripts de soumission et chemin de sortie du runner sont adaptés. Les empreintes des références100 historiques sont conservées pour chaque graine. Versions, threads, sources et fichiers d'entrée vérifiés en prévol ; cible/masque/régions vérifiés avant entraînement ; entrées, initialisations et paramètres vérifiés après entraînement.

Allocation : array0–3, deux tâches simultanées,4CPU,16Go,36h chacune. Temps attendu de l'ordre20–24h par tâche d'après les pilotes, hors attente, sans garantie. Les tâches historiques300 restent indépendantes.

Analyse prévue après rapatriement vérifié : trois graines par souris, reconstruction globale/régionale, courants par unité, PCA10–CCA multidimensionnelle, contrôle de source avec axes appris/testés sur blocs temporels distincts, amplitudes séparées. Les graines sont des répétitions techniques. Le réseau apprend la session entière ; le découpage de CCA ne constitue pas une validation du réseau sur une session inédite. Aucune conclusion sur l'âge à cette étape.

État initial : configurations préparées, soumission en attente de connexion ; seul un reçu submission.jobid et la vérification Slurm confirmeront la soumission.

## Soumission confirmée

Job array **3178099**, quatre tâches acceptées par Slurm après réussite des tests et des quatre prévols. Reçu et journaux rapatriés dans ce dossier. Le numéro3178098 affiché par --test-only n'est pas le job soumis. Les contrôles d'identité de la cible prétraitée s'effectuent au démarrage effectif des entraînements (le prévol dry-run ne charge pas les TIFF).
