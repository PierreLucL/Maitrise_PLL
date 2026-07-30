#### Importations et fonctions pour l'analyse des données GCaMP ####

import numpy as np
from sklearn.cluster import KMeans
from scipy import ndimage
from scipy.spatial.distance import cdist
from collections import defaultdict
from scipy import ndimage as ndi

#### DIMINUER LES RÉGIONS DU ALLEN ATLAS

ATLAS_TO_PARENT_6 = {
    # ============================================================
    # 0 — Rég. M.II
    # Ensemble frontal / moteur secondaire
    # ============================================================
    1: 0,    # ACAd_left
    2: 0,    # AUDd_left
    3: 0,    # AUDp_left
    4: 0,    # AUDpo_left
    5: 0,    # AUDv_left
    6: 0,    # ECT_left
    7: 0,    # FRP_left
    9: 0,    # MOs_left
    10: 0,   # ORBm_left
    11: 0,   # PL_left

    35: 0,   # ACAd_right
    36: 0,   # AUDd_right
    37: 0,   # AUDp_right
    38: 0,   # AUDpo_right
    39: 0,   # AUDv_right
    40: 0,   # ECT_right
    41: 0,   # FRP_right
    43: 0,   # MOs_right
    44: 0,   # ORBm_right
    45: 0,   # PL_right

    # ============================================================
    # 1 — Rég. M.I
    # Cortex moteur primaire
    # ============================================================
    8: 1,    # MOp_left
    42: 1,   # MOp_right

    # ============================================================
    # 2 — Rég. Som.
    # Cortex somatosensoriel primaire et secondaire
    # + aire viscérale
    # ============================================================
    15: 2,   # SSp-bfd_left
    16: 2,   # SSp-ll_left
    17: 2,   # SSp-m_left
    18: 2,   # SSp-n_left
    19: 2,   # SSp-tr_left
    20: 2,   # SSp-ul_left
    21: 2,   # SSp-un_left
    22: 2,   # SSs_left
    26: 2,   # VISC_left — visceral area

    49: 2,   # SSp-bfd_right
    50: 2,   # SSp-ll_right
    51: 2,   # SSp-m_right
    52: 2,   # SSp-n_right
    53: 2,   # SSp-tr_right
    54: 2,   # SSp-ul_right
    55: 2,   # SSp-un_right
    56: 2,   # SSs_right
    60: 2,   # VISC_right — visceral area

    # ============================================================
    # 3 — Rég. Ass.
    # Auditif / temporal / associatif latéral
    # ============================================================
    31: 3,   # VISa_left
    34: 3,   # VISrl_left

    65: 3,   # VISa_right
    68: 3,   # VISrl_right

    # ============================================================
    # 4 — Rég. Vis.
    # Aires visuelles primaires et associatives
    # ============================================================
    23: 4,   # TEa_left
    24: 4,   # VISal_left
    25: 4,   # VISam_left
    27: 4,   # VISl_left
    28: 4,   # VISp_left
    29: 4,   # VISpl_left
    30: 4,   # VISpm_left
    32: 4,   # VISli_left
    33: 4,   # VISpor_left

    57: 4,   # TEa_right
    58: 4,   # VISal_right
    59: 4,   # VISam_right
    61: 4,   # VISl_right
    62: 4,   # VISp_right
    63: 4,   # VISpl_right
    64: 4,   # VISpm_right
    66: 4,   # VISli_right
    67: 4,   # VISpor_right

    # ============================================================
    # 5 — Rég. Rét.
    # Cortex rétrosplénial
    # ============================================================
    12: 5,   # RSPagl_left
    13: 5,   # RSPd_left
    14: 5,   # RSPv_left

    46: 5,   # RSPagl_right
    47: 5,   # RSPd_right
    48: 5,   # RSPv_right
}

REGION_NAMES_6 = {
    0: "Frontal_MoteurSecondaire",
    1: "MoteurPrimaire",
    2: "Somatosensoriel",
    3: "Associatif_Auditif_Temporal",
    4: "Visuel",
    5: "Retrosplenial",
}

def remove_thin_label_artifacts(
    atlas,
    size=3,
    min_fraction=0.3,
    background=np.nan
):
    """
    Remplace uniquement les pixels dont le label est peu représenté
    dans leur voisinage local.
    """

    atlas = np.asarray(atlas)
    result = atlas.copy()

    if np.isnan(background):
        valid = ~np.isnan(atlas)
    else:
        valid = atlas != background

    labels = np.unique(atlas[valid])

    local_counts = []

    for label in labels:
        count = ndi.uniform_filter(
            (atlas == label).astype(float),
            size=size,
            mode="nearest"
        )
        local_counts.append(count)

    local_counts = np.stack(local_counts, axis=0)

    majority_index = np.argmax(local_counts, axis=0)
    majority_label = labels[majority_index]

    # Fraction du voisinage appartenant au label actuel
    current_fraction = np.zeros(atlas.shape, dtype=float)

    for i, label in enumerate(labels):
        pixels = atlas == label
        current_fraction[pixels] = local_counts[i][pixels]

    suspicious = valid & (current_fraction < min_fraction)

    result[suspicious] = majority_label[suspicious]

    return result

def reduce_atlas_to_6_regions(
    atlas: np.ndarray,
    roi_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Regroupe les 68 régions Allen en 6 régions bilatérales.

    Paramètres
    ----------
    atlas
        Atlas 2D contenant les labels 1 à 68.

    roi_mask
        Masque 2D optionnel. Les pixels à 0 seront placés à NaN.

    Retour
    ------
    atlas_6
        Atlas 2D contenant les labels 0 à 5 et NaN pour le fond.
    """
    if atlas.ndim != 2:
        raise ValueError(
            f"L'atlas doit être 2D, mais sa shape est {atlas.shape}."
        )

    if roi_mask is not None and roi_mask.shape != atlas.shape:
        raise ValueError(
            f"Dimensions incompatibles : "
            f"atlas={atlas.shape}, roi_mask={roi_mask.shape}."
        )

    atlas_6 = np.full(atlas.shape, np.nan, dtype=float)

    for original_label, parent_label in ATLAS_TO_PARENT_6.items():
        atlas_6[atlas == original_label] = parent_label

    if roi_mask is not None:
        atlas_6[roi_mask == 0] = np.nan

    return atlas_6

def clean_reduced_atlas(
    atlas_6,
    brain_mask=None,
    min_component_size=100,
    tie_tolerance=1.0
):
    """
    Nettoie un atlas réduit à 6 régions.

    Paramètres
    ----------
    atlas_6 : array 2D
        Masque contenant les labels 0 à 5.
        Le fond peut être NaN ou une autre valeur.

    brain_mask : array bool 2D, optionnel
        Zone dans laquelle les pixels doivent appartenir à une région.
        Fortement recommandé pour ne pas remplir l'extérieur du cerveau.

    min_component_size : int
        Taille minimale d'une composante considérée comme fiable.

    tie_tolerance : float
        Tolérance en pixels. Parmi les régions dont la distance est à moins
        de cette valeur de la distance minimale, la plus grande est choisie.
    """

    atlas_6 = np.asarray(atlas_6, dtype=float)

    valid_labels = np.arange(6)

    # Pixels ayant déjà un label valide
    valid = np.isin(atlas_6, valid_labels)

    if brain_mask is None:
        # Solution de repli : seulement la zone déjà occupée par l'atlas.
        # Mieux vaut fournir un vrai masque du cerveau enregistré.
        brain_mask = ndi.binary_closing(
            valid,
            structure=np.ones((5, 5)),
            iterations=2
        )
        brain_mask = ndi.binary_fill_holes(brain_mask)
    else:
        brain_mask = np.asarray(brain_mask, dtype=bool)

    trusted = np.full(atlas_6.shape, np.nan)
    region_sizes = np.zeros(6, dtype=int)

    structure = np.ones((3, 3), dtype=int)

    # Conserver les grandes composantes connexes de chaque région
    for region in valid_labels:
        region_mask = atlas_6 == region
        components, n_components = ndi.label(
            region_mask,
            structure=structure
        )

        if n_components == 0:
            continue

        sizes = np.bincount(components.ravel())
        sizes[0] = 0

        keep_ids = np.where(sizes >= min_component_size)[0]

        # Toujours conserver au moins la composante principale
        if len(keep_ids) == 0:
            keep_ids = [np.argmax(sizes)]

        keep_mask = np.isin(components, keep_ids)

        trusted[keep_mask] = region
        region_sizes[region] = np.sum(keep_mask)

    trusted_valid = np.isin(trusted, valid_labels)

    # Distance de chaque pixel au noyau fiable de chaque région
    distance_maps = np.full((6, *atlas_6.shape), np.inf)

    for region in valid_labels:
        seed = trusted == region

        if np.any(seed):
            distance_maps[region] = ndi.distance_transform_edt(~seed)

    min_distance = np.min(distance_maps, axis=0)

    # Une région est candidate si elle est presque aussi proche que la meilleure
    candidates = (
        distance_maps
        <= min_distance[None, :, :] + tie_tolerance
    )

    # Parmi les candidates, sélectionner la plus grande région
    candidate_sizes = np.where(
        candidates,
        region_sizes[:, None, None],
        -1
    )

    selected_region = np.argmax(candidate_sizes, axis=0)

    result = trusted.copy()

    pixels_to_reassign = brain_mask & ~trusted_valid
    result[pixels_to_reassign] = selected_region[pixels_to_reassign]

    # Préserver l'extérieur du cerveau
    result[~brain_mask] = np.nan

    return result

### CLEANER LE MASQUE DE REGIONS ###

def extract_nonzero_pixels(dataset,debug=False):
    """
    Extracts pixels from the dataset that have at least one non-zero value.

    Args:
        dataset: array (T, H, W) float32

    Returns:
        active_pixels: array (N, T) float32, where N is the number of active pixels
        mask: array (H, W) bool
    """

    X = dataset
    T, H, W = X.shape

    # reshape -> (pixels, temps)
    pixels = X.reshape(T, H*W).T   # (H*W, T)

    # Delta F sur F
    F0 = np.percentile(pixels, 8, axis=1, keepdims=True)
    dff = (pixels - F0) / (F0 + 1e-8)

    # garder pixels dont au moins une valeur > 0
    active_mask = np.max(dff, axis=1) > 0

    active_pixels = pixels[active_mask]
    mask = active_mask.reshape(H, W)

    if debug:
        print(f"Total pixels : {H*W}")
        print(f"Pixels non noirs : {active_pixels.shape[0]}")
        print(f"Shape finale : {active_pixels.shape}")

    return active_pixels, mask

def clean_region_mask(region_mask, min_component_size=50, background_value=np.nan):
    """
    Nettoie le masque de régions en réassignant les petites composantes connexes
    isolées à la région voisine la plus proche (par dilatation successive).

    En plus :
    - la région parent d'indice 0.0 est transformée en fond (NaN)
    - les autres labels sont décalés de -1 :
        1.0 -> 0.0
        2.0 -> 1.0
        etc.

    Paramètres
    ----------
    region_mask : ndarray (H, W)
    min_component_size : int
        Toute composante connexe d'une région avec moins de pixels que ce seuil
        sera réassignée à la région voisine la plus proche.
    background_value : float
        Valeur du fond (NaN ou numérique).

    Retour
    ------
    cleaned_mask : ndarray (H, W)
    """
    region_mask = np.asarray(region_mask, dtype=float)

    # Transformer la région 0 en fond, puis décaler les autres labels
    region_mask = np.where(region_mask == 0, np.nan, region_mask - 1)

    cleaned = region_mask.copy()
    use_nan_bg = np.isnan(background_value)
    valid = np.isfinite(cleaned) if use_nan_bg else (cleaned != background_value)
    region_labels = np.unique(cleaned[valid])

    # Identifier toutes les petites composantes à corriger
    to_fix = np.zeros(region_mask.shape, dtype=bool)
    for label in region_labels:
        cc_map, n_cc = ndimage.label(cleaned == label)
        for cc_id in range(1, n_cc + 1):
            if (cc_map == cc_id).sum() < min_component_size:
                to_fix |= (cc_map == cc_id)

    # Réassigner par dilatation successive : on propage les labels voisins
    # valides dans les pixels à corriger, jusqu'à ce que tous soient couverts
    cleaned[to_fix] = np.nan
    remaining = to_fix.copy()

    while remaining.any():
        # Dilater le masque des pixels valides d'un pixel (4-connexe)
        valid_now = np.isfinite(cleaned)
        dilated = ndimage.binary_dilation(
            valid_now,
            structure=ndimage.generate_binary_structure(2, 1)
        )
        newly_covered = dilated & remaining

        # Assigner à chaque pixel nouvellement couvert le label de son voisin valide
        for r, c in np.argwhere(newly_covered):
            neighbors = [cleaned[r + dr, c + dc]
                         for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                         if 0 <= r + dr < cleaned.shape[0]
                         and 0 <= c + dc < cleaned.shape[1]
                         and np.isfinite(cleaned[r + dr, c + dc])]
            if neighbors:
                cleaned[r, c] = max(set(neighbors), key=neighbors.count)
                remaining[r, c] = False

    return cleaned

def remove_dead_pixels_from_region_mask(region_mask,
                                        pixel_mask,
                                        pixel_mask_is_dead=False,
                                        background_value=np.nan,
                                        return_removed_mask=False):
    """
    Retire les pixels morts d'un masque de régions.

    Paramètres
    ----------
    region_mask : ndarray (H, W)
        Masque des régions. Peut contenir des labels numériques et un fond.
    pixel_mask : ndarray (H, W), bool
        Masque booléen :
        - si pixel_mask_is_dead=True : True = pixel mort
        - si pixel_mask_is_dead=False : True = pixel vivant
    pixel_mask_is_dead : bool
        Indique comment interpréter pixel_mask.
    background_value :
        Valeur à mettre aux pixels exclus. Peut être np.nan.
    return_removed_mask : bool
        Si True, retourne aussi un masque booléen des pixels retirés.

    Retour
    ------
    cleaned_region_mask : ndarray (H, W)
        Copie de region_mask où les pixels morts ont été remplacés par le fond.
    removed_mask : ndarray (H, W), bool   [optionnel]
        True là où un pixel de région a été retiré car mort.
    """

    region_mask = np.asarray(region_mask)
    pixel_mask = np.asarray(pixel_mask)

    if region_mask.shape != pixel_mask.shape:
        raise ValueError("region_mask et pixel_mask doivent avoir la même shape.")

    if pixel_mask.dtype != bool:
        pixel_mask = pixel_mask.astype(bool)

    # Interprétation du masque booléen
    if pixel_mask_is_dead:
        dead_mask = pixel_mask
    else:
        dead_mask = ~pixel_mask

    cleaned_region_mask = region_mask.copy()

    # Détecter les pixels qui appartiennent réellement à une région
    if np.issubdtype(region_mask.dtype, np.floating) and np.isnan(background_value):
        in_region = np.isfinite(region_mask)
    else:
        in_region = region_mask != background_value

    # Pixels à retirer = pixels dans une région ET morts
    removed_mask = in_region & dead_mask

    # Si on veut mettre des NaN, il faut un dtype float
    if np.issubdtype(type(background_value), np.floating) and np.isnan(background_value):
        cleaned_region_mask = cleaned_region_mask.astype(float)

    cleaned_region_mask[removed_mask] = background_value

    if return_removed_mask:
        return cleaned_region_mask, removed_mask
    return cleaned_region_mask

### SUBDIVISER EN SOUS-SECTIONS ###

def subdivide_mask_by_spatial_clustering(region_mask,
                                         target_size=100,
                                         background_value=np.nan,
                                         random_state=0,
                                         n_init=10):
    region_mask = np.asarray(region_mask, dtype=float)
    H, W = region_mask.shape

    use_nan_bg = np.isnan(background_value)
    subgroup_mask = np.full((H, W), np.nan if use_nan_bg else background_value, dtype=float)
    valid = np.isfinite(region_mask) if use_nan_bg else (region_mask != background_value)
    region_labels = np.unique(region_mask[valid])

    subgroup_info = {}
    global_id = 0

    for region_label in region_labels:
        coords = np.argwhere(region_mask == region_label)
        N = len(coords)
        n_clusters = max(1, min(int(np.ceil(N / target_size)), N))

        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
        km.fit(coords)
        local_labels = np.argmin(cdist(coords, km.cluster_centers_), axis=1)

        # Absorber les pixels isolés (composante connexe de taille 1)
        # dans leur voisin immédiat le plus fréquent
        label_map = np.full((H, W), -1, dtype=int)
        label_map[coords[:, 0], coords[:, 1]] = local_labels

        changed = True
        while changed:
            changed = False
            for loc_lab in np.unique(label_map[label_map >= 0]):
                cc_map, n_cc = ndimage.label(label_map == loc_lab)
                for cc_id in range(1, n_cc + 1):
                    cc_pixels = np.argwhere(cc_map == cc_id)
                    if len(cc_pixels) > 1:
                        continue
                    r, c = cc_pixels[0]
                    neighbors = [label_map[r+dr, c+dc]
                                 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                                 if 0 <= r+dr < H and 0 <= c+dc < W
                                 and label_map[r+dr, c+dc] >= 0]
                    if neighbors:
                        label_map[r, c] = max(set(neighbors), key=neighbors.count)
                        changed = True

        local_labels = label_map[coords[:, 0], coords[:, 1]]

        for loc_lab in np.unique(local_labels):
            chunk = coords[local_labels == loc_lab]
            subgroup_mask[chunk[:, 0], chunk[:, 1]] = global_id
            subgroup_info[global_id] = {
                "parent_region": region_label,
                "local_subgroup_id": int(loc_lab),
                "n_pixels": len(chunk),
                "centroid": tuple(chunk.mean(axis=0).tolist())
            }
            global_id += 1

    # Réordonnancement haut-droite → bas-gauche
    sorted_ids = sorted(subgroup_info, key=lambda i: (
        subgroup_info[i]["centroid"][0],
        -subgroup_info[i]["centroid"][1]
    ))
    old_to_new = {old: new for new, old in enumerate(sorted_ids)}
    new_mask = subgroup_mask.copy()
    for old, new in old_to_new.items():
        new_mask[subgroup_mask == old] = new

    return new_mask, {new: subgroup_info[old] for old, new in old_to_new.items()}

#### CONSTRUIRE LE DICTIONNAIRE DES REGIONS PARENTS ###

def build_parent_regions_dict(info_masque_sub):
    """
    Retourne :
    regions = array de shape (n_regions, 2) dtype=object, où chaque ligne est :
    [nom_region, array(indices_sous_regions)]
     - nom_region : string "Region A", "Region B", etc.
     - indices_sous_regions : array 1D des indices des sous-régions appartenant à cette région
     - les régions sont triées par ordre croissant d'indice de région parente (A=0, B=1, etc.)
     - les indices des sous-régions dans chaque région sont triés par ordre croissant d'indice de sous-région
     - n_regions = nombre de régions parentes (ex: 3 pour A, B, C)
     - les indices de sous-régions sont ceux utilisés dans masque_sub et info_masque_sub
     - les indices de sous-régions sont uniques (une sous-région n'appartient qu'à une seule région parente)
     - les régions parentes sont celles indiquées dans info_masque_sub['parent_region']
    """
    regions = defaultdict(list)
    region_names = {
    0: "Rég. M.II",
    1: "Rég. M.I",
    2: "Rég. Som.",
    3: "Rég. Ass.",
    4: "Rég. Vis.",
    5: "Rég. Rét."}

    for idx, info in info_masque_sub.items():
        regions[int(info['parent_region'])].append(idx)

    regions = np.array(
        [[region_names.get(r, f"Région {r}"), np.array(indices)]
        for r, indices in sorted(regions.items())],
        dtype=object)

    return regions
