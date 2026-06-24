#### Importations et fonctions pour l'analyse des données GCaMP ####

import numpy as np
from sklearn.cluster import KMeans
from scipy import ndimage
from scipy.spatial.distance import cdist
from collections import defaultdict

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
