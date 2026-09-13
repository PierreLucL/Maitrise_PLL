# Sauvegarde GitHub du 13 septembre 2026

Ce point de sauvegarde conserve la réorganisation du dépôt, le pipeline configurable, les scripts d’analyse et de visualisation, les tests, les configurations et scripts Narval, le contexte scientifique consolidé, les bilans, figures et résultats tabulaires disponibles. Il inclut les copies figées du lot de contrôles 2989725, leur manifeste et le reçu de soumission.

Validation avant commit : 18 tests unitaires réussis ; état local synchronisé avec origin/main avant la sauvegarde ; absence de fichiers supérieurs à 50 Mo et de modèles binaires dans les ajouts. Les espaces historiques dans les archives, CSV et copies figées sont conservés pour ne pas altérer leur provenance.

GitHub ne contient pas les données expérimentales ni les modèles PKL et caches NPZ. L’inventaire `inventaire_fichiers_lourds_2026-09-13.json` liste 329 fichiers PKL/NPZ locaux avec chemin, taille et SHA-256. Un inventaire ne remplace pas leur sauvegarde. Les modèles rapatriés restent locaux et leurs références distantes sont documentées dans les bilans ; leur rétention future sur scratch Narval n’est pas garantie par ce push. Les sources expérimentales restent sur le disque de données et Narval.

Les dossiers `results/` suivent désormais explicitement les formats légers de rapports, figures, tables, provenance, scripts figés et reçus. Les PKL, NPZ, TIFF, NPY, H5, MAT et archives ZIP restent ignorés.
