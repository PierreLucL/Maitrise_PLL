# Inventaire longitudinal — 11 septembre 2026

Mise en contexte : cinq dossiers incomplets (C0/M20/42, C2/M6/308, C5/M12/353, C5/M14/353, C9/M16/410) avaient déjà été exclus sur demande de Pierre-Luc dans les discussions d’août. Ils ne sont pas de nouvelles découvertes ni des candidats automatiquement réactivés. Le masque manquant de C2/M8/308 constitue une divergence supplémentaire du scan actuel par rapport aux 58 dossiers complets annoncés historiquement ; sa cause n’est pas établie. Voir ../../docs/contexte_discussions_maitrise.md.

## Portée et provenance

Scan des dossiers du disque local et de Narval, sans modifier les données. Les inventaires JSON adjacents contiennent les chemins, noms et tailles des fichiers et la date UTC du scan. Un dossier est « complet » si GCaMP.tif, atlas.npy et roi_mask.tif sont présents et non vides. Cela ne valide ni leur contenu ni leur qualité.

- Local : 63 dossiers, 57 complets, 23 souris avec au moins un dossier complet.
- Narval : 58 dossiers, 57 complets.
- Catalogue du code : 58 entrées. Entrées déclarées mais incomplètes localement : [(2, 8, 308)].

M6, M8, etc. sont ici des étiquettes de dossiers. Âges exacts, dates, sexe, état comportemental, identifiants de session et continuité de l’identité des souris restent à confirmer avec les métadonnées. Un dossier ne prouve pas une session biologique unique.

## Couverture des dossiers complets

| Cohorte | Souris | Mois disponibles | Nombre |
|---|---|---|---|
| C0 | 191 | M12 | 1 |
| C0 | 210 | M12 | 1 |
| C0 | 213 | M12 | 1 |
| C0 | 233 | M12 | 1 |
| C0 | 253 | M10, M20 | 2 |
| C2 | 304 | M20 | 1 |
| C2 | 308 | M10, M12, M14, M16 | 4 |
| C3 | 316 | M6, M8, M10, M12, M14, M16, M18, M20 | 8 |
| C3 | 322 | M6, M8, M10, M12, M14 | 5 |
| C5 | 353 | M6, M8, M10 | 3 |
| C5 | 359 | M18, M20 | 2 |
| C5 | 361 | M6 | 1 |
| C6 | 365 | M6 | 1 |
| C6 | 367 | M6 | 1 |
| C6 | 374 | M6, M8, M10, M16, M18 | 5 |
| C7 | 387 | M6, M10, M12 | 3 |
| C7 | 396 | M6 | 1 |
| C7 | 397 | M6 | 1 |
| C8 | 408 | M18, M20 | 2 |
| C8 | 409 | M6 | 1 |
| C9 | 410 | M6, M8, M10, M12, M14 | 5 |
| C9 | 412 | M8, M10, M12 | 3 |
| C9 | 415 | M6, M8, M18, M20 | 4 |

## Dossiers incomplets

| Dossier | Fichiers manquants | Présent sur Narval |
|---|---|---|
| C0_M20/Data/RS_M42 | atlas.npy, roi_mask.tif | non |
| C2_M6/Data/RS_M308 | atlas.npy, roi_mask.tif | non |
| C2_M8/Data/RS_M308 | roi_mask.tif | oui |
| C5_M12/Data/RS_M353 | atlas.npy, roi_mask.tif | non |
| C5_M14/Data/RS_M353 | atlas.npy, roi_mask.tif | non |
| C9_M16/Data/RS_M410 | atlas.npy, roi_mask.tif | non |

## Panel pilote proposé, sous réserve du contrôle qualité

- Suivis longs : C3/souris316 à M6, M12, M18 ; C6/souris374 à M6, M10, M18 ; C9/souris415 à M6, M8, M18. Neuf dossiers au total, avec les trois fichiers requis sur Narval.
- Compléments : C9/souris410 à M6, M8, M14 et C3/souris322 à M6, M12, M14. Ils ajoutent des comparaisons entre souris de la même cohorte et des suivis intermédiaires. Le M6 de 410/415 bénéficie déjà des trois graines calculées.
- Ce panel exploratoire ne suffit pas à une estimation robuste de l’effet de l’âge : trois souris suivies tardivement, chacune d’une cohorte différente, et couverture inégale des âges. Ne pas confondre souris, cohortes et répétitions techniques.
- Pour un premier contraste longitudinal comparable en étiquettes : examiner M6 → M18 chez 316, 374 et 415 (six dossiers), avec les trois âges intermédiaires ci-dessus. M20 est également disponible pour 316 et 415.

## Prochaines décisions

1. Contrôler atlas/masques, dimensions, cadence réelle, durée et qualité des neuf dossiers M6/M18 plus un âge intermédiaire par souris.
2. Retrouver les métadonnées biologiques et d’acquisition. Le fichier Datasets.numbers existe sur le disque mais son contenu n’a pas été interprété dans ce scan.
3. Définir les caractéristiques des courants à partir du bilan entre graines, puis arrêter la normalisation et les contrôles de spécificité avant de lancer le panel.
4. Examiner la possibilité de compléter les atlas/masques manquants, sans les fabriquer ou copier automatiquement depuis une autre session.

## Vérification dimensionnelle du panel

Les neuf dossiers du panel contiennent chacun 5 760 images GCaMP. Les dimensions spatiales de GCaMP, de l’atlas et du masque concordent au sein de chacun des neuf dossiers. Les dimensions varient entre sessions ; cette vérification ne garantit pas la correspondance anatomique. Cadence réelle et durée en secondes restent à confirmer. Détails : `pilot_dimensions.json`.
