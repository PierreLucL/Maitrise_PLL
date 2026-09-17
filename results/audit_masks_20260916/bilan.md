# Audit des masques et des conventions — 16 septembre 2026

Périmètre : src/maitrise_curbd/masks.py, prepare_timeseries dans scripts/curbd/loop.py, extract_timeseries_du_tenseur ; masques sauvegardés des pilotes tt-1 410/415 graine2026. Aucun changement du moteur, de segmentation ou des tâches Narval.

## Chaîne réellement exécutée

Chargement atlas/ROI → remove_thin_label_artifacts (fenêtre5×5, seuil0,25) → reduce_atlas_to_6_regions (ROI appliqué ici) → KMeans spatial par région parente (taille cible15) → build_parent_regions_dict → moyenne des pixels finis par parcelle.

clean_reduced_atlas, clean_region_mask, remove_dead_pixels_from_region_mask et extract_nonzero_pixels ne sont pas appelées par cette chaîne. Les commentaires des anciennes explorations sur les pixels morts ne décrivent donc pas le traitement actuel.

## Constats prioritaires

1. **Les noms anatomiques ne décrivent pas fidèlement la table.** Selon les commentaires associés aux labels numériques, M.II contient aussi AUDd/p/po/v, ECT, ACAd, FRP, ORBm, PL ; Ass. contient VISa/VISrl, malgré REGION_NAMES_6 qui la nomme Associatif_Auditif_Temporal ; Vis. contient TEa. Les groupes sont bilatéraux. Vérifier la légende originale d'atlas.npy : les entiers1–68 sont une convention locale, et cet audit du code ne certifie pas leur correspondance anatomique. REGION_NAMES_6 n'est pas utilisée par build_parent_regions_dict, qui maintient sa propre table de noms.
2. **Pas de remplissage systématique des pixels non attribués.** Les NaN sont conservés par remove_thin_label_artifacts ; les labels non mappés deviennent NaN à la réduction ; ils sont exclus des parcelles et des moyennes. Un fond codé0 est toutefois traité comme un label pendant le nettoyage initial et peut être converti en voisin connu. Les conventions0/NaN ont donc un effet réel. Le nettoyage a lieu avant application du ROI : des pixels hors ROI participent au voisinage des pixels intérieurs.
3. **Le nettoyage modifie des labels anatomiques valides.** Tout pixel dont le label occupe moins de25% de sa fenêtre5×5 passe au label le plus fréquent ; ce n'est ni un test de composante isolée ni une détection biologique d'artefact. Les NaN occupent une partie du dénominateur du voisinage. Une bordure fine peut être modifiée.
4. **pixels15 est une cible de taille, pas15 pixels garantis ni15×15.** Le nombre initial de clusters est ceil(nombre de pixels du parent/15), avec KMeans sur les coordonnées, sans utiliser l'activité. Le correctif absorbe seulement les composantes de taille1 ayant un voisin direct ; aucune garantie générale de connexité. Les IDs sont ensuite classés par centroïde, ligne croissante puis colonne décroissante ; ils ne constituent pas des identifiants anatomiques homologues entre sessions.
5. **Le dictionnaire est en réalité un tableau objet (région, IDs des unités).** Les IDs sont cohérents dans les deux sauvegardes examinées. La docstring annonce un tri des IDs, mais la fonction conserve leur ordre d'insertion ; la chaîne actuelle les fournit déjà dans le bon ordre. Des IDs non contigus venant d'un autre appel pourraient désaligner le tableau avec l'extraction des traces, qui compacte les labels triés. Une région absente est omise, sans ligne vide ni parent_id explicite.

## Contrôle des résultats existants

| Souris | Unités | Taille min / médiane / max | Parcelles non connexes (4-voisins) |
|---|---:|---|---:|
|410|2538|4 /15 /24|3|
|415|2331|2 /15 /25|8|

IDs contigus0..N−1 ; couverture unique de tous les IDs dans regions ; parents cohérents ; n_pixels exact ; dimensions Adata/RNN cohérentes. Pas de décalage d'index détecté dans ces deux modèles. Une parcelle non connexe à4voisins peut être connexe en diagonale : ne pas assimiler ce chiffre à un mélange entre régions ou hémisphères. La validation ne prouve pas l'alignement anatomique entre acquisitions.

## Fonctions alternatives à ne pas activer telles quelles

- clean_reduced_atlas : réattribue par distance euclidienne, avec préférence pour la région la plus grande parmi les candidates à moins de tie_tolerance de la meilleure distance, sans plafond de distance. Les noyaux sont construits avant restriction au brain_mask. Sans aucun noyau, elle attribue actuellement0 partout dans le brain_mask (reproduit synthétiquement), ce qui invente une région.
- clean_region_mask : transforme0 en fond et décale les autres labels ; incompatible avec atlas_6 où0 est une vraie région. Sa boucle peut rester bloquée si remaining contient des pixels qu'aucun noyau valide ne peut atteindre par les seuls pixels à corriger.
- extract_nonzero_pixels : son nom et sa docstring ne décrivent pas son calcul ; utilise un percentile puis une transformation de type ΔF/F pour sélectionner des pixels. Ne pas la réintroduire implicitement sur les données déjà ΔF/F.

## Suites proposées

Confirmer d'abord les six regroupements anatomiques voulus et la légende source des labels. Documenter explicitement exclusion ou réattribution du fond ; rapporter la couverture et les pixels modifiés par région/session. Ajouter des garde-fous de labels/IDs/ROI, journaliser tailles et connexité ; distinguer ces garde-fous d'une modification de segmentation. Toute modification qui change les parcelles doit être une nouvelle version contrôlée, pour éviter de confondre méthode et âge. Les réplications en cours restent comparables aux pilotes puisqu'elles utilisent les snapshots figés.

Contrôles synthétiques reproductibles : scripts/analysis/audit_mask_contracts.py ; sorties synthetic_checks.json. Mesures sur sauvegardes : saved_masks_audit.json.
