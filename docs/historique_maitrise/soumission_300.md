# Soumission 410, graine 2026, 300 passages

Job Slurm **2796272_0**, soumis le 10 septembre 2026. 4 CPU, 16 Gio, limite 96 h. État à la soumission : en attente. Aucun doublon créé.

Le calcul utilise directement le snapshot figé `/scratch/pllar11/Maitrise_PLL/run_snapshots/repro15_b3abdc905a6b5302` des six entraînements à 100 passages. Configuration identique pour C9/M6/410, graine 2026, sauf nRunTrain=300. L’indice historique tt est conservé. Les empreintes du snapshot, présence des données, tests et validation sbatch ont passé avant soumission.

Configuration et lanceur : `/scratch/pllar11/Maitrise_PLL/run_snapshots/train300_410_2026`. Reçu distant : `submission.jobid`. Résultats attendus : `/scratch/pllar11/Maitrise_PLL/results/narval_train300_410_2026/2796272/`.

Estimation de calcul grossière : environ 69 h à partir des 23 h observées ; 96 h est une limite avec marge, pas une durée prévue. L’attente Slurm s’ajoute et peut être longue pour cette classe de durée. Lors du test préalable Slurm indiquait le 29 septembre ; cette projection n’est pas une garantie de démarrage.
