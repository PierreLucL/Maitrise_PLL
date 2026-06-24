import h5py
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from Pipeline import *

def extract_pixel_timeseries_per_region(
    X,
    mask,
    inactive_mask=None,
    n_regions=7,
    labels=None,
    nan_safe=True,
    copy=True):
    """
    X: (T, H, W) float, peut contenir NaN/inf
    mask: (H, W) float avec labels (0..6) et NaN pour fond
    inactive_mask: (H, W) bool ou 0/1
        True = pixel inactif (à exclure)
        False = pixel actif (à garder)

    Retour:
      regions_ts: list de longueur n_regions
        regions_ts[r] = list de arrays shape (T,) (une timeseries par pixel actif)
      pixel_indices: list de longueur n_regions
        pixel_indices[r] = array shape (Npix_r, 2) avec (y, x) des pixels actifs
    """

    if labels is None:
        labels = list(range(n_regions))
    else:
        labels = list(labels)
        n_regions = len(labels)

    X = np.asarray(X)
    mask = np.asarray(mask)

    if X.ndim != 3:
        raise ValueError(f"X doit être (T,H,W), reçu shape={X.shape}")
    if mask.ndim != 2:
        raise ValueError(f"mask doit être (H,W), reçu shape={mask.shape}")

    T, H, W = X.shape
    if mask.shape != (H, W):
        raise ValueError("mask incompatible avec X")

    # --- Gestion du inactive_mask ---
    if inactive_mask is None:
        inactive_mask = np.zeros((H, W), dtype=bool)
    else:
        inactive_mask = np.asarray(inactive_mask).astype(bool)
        if inactive_mask.shape != (H, W):
            raise ValueError("inactive_mask incompatible avec X")

    # --- Discrétisation des labels ---
    valid = np.isfinite(mask)
    m = np.full((H, W), -1, dtype=np.int32)
    m[valid] = np.rint(mask[valid]).astype(np.int32)

    # Aplatit
    m_flat = m.reshape(-1)
    inactive_flat = inactive_mask.reshape(-1)

    X_flat = X.reshape(T, H * W)

    regions_ts = []
    pixel_indices = []

    for lab in labels:

        # Pixels de la région ET actifs
        pix = np.flatnonzero(
            (m_flat == int(lab)) & (~inactive_flat)
        )

        if pix.size == 0:
            regions_ts.append([])
            pixel_indices.append(np.empty((0, 2), dtype=int))
            continue

        ys, xs = np.divmod(pix, W)
        pixel_indices.append(np.stack([ys, xs], axis=1))

        block = X_flat[:, pix]  # (T, Npix)

        if nan_safe:
            block = np.where(np.isfinite(block), block, np.nan)

        if copy:
            regions_ts.append(
                [block[:, j].astype(np.float64, copy=True)
                 for j in range(block.shape[1])]
            )
        else:
            regions_ts.append(
                [block[:, j] for j in range(block.shape[1])]
            )

    return regions_ts, pixel_indices

def remove_dead_pixels(regions_ts, pixel_indices=None, tol=1e-12):
    """
    Enlève les time series complètement nulles (ou quasi-nulles).
    
    regions_ts: list de listes d'arrays (T,)
    pixel_indices: list de arrays (Npix,2) optionnel
    
    tol: seuil numérique
    
    Retour:
        cleaned_regions_ts
        cleaned_pixel_indices (si fourni)
    """
    cleaned_regions_ts = []
    cleaned_pixel_indices = [] if pixel_indices is not None else None

    for r in range(len(regions_ts)):

        new_ts_list = []
        new_coords = []

        for i, ts in enumerate(regions_ts[r]):

            # Condition pixel mort
            if (np.nanmax(ts) - np.nanmin(ts)) > tol:
                new_ts_list.append(ts)

                if pixel_indices is not None:
                    new_coords.append(pixel_indices[r][i])

        cleaned_regions_ts.append(new_ts_list)

        if pixel_indices is not None:
            if len(new_coords) > 0:
                cleaned_pixel_indices.append(np.vstack(new_coords))
            else:
                cleaned_pixel_indices.append(np.empty((0, 2), dtype=int))

    if pixel_indices is not None:
        return cleaned_regions_ts, cleaned_pixel_indices
    else:
        return cleaned_regions_ts
    

def plot_region_timeseries(ts, labels):
    """
    Affiche les time series moyennées par région.
    ts: (n_regions, T)
    labels: (n_regions,)
    """
    T = ts.shape[1]
    t = np.arange(T)
    n = ts.shape[0]

    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(10, 7))
    if n == 1: axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t, ts[i], linewidth=1)
        ax.set_ylabel(f"{int(labels[i])}", rotation=0, labelpad=20)
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (frame)")
    fig.suptitle("Time series par région (moyenne des pixels)")
    plt.tight_layout()
    plt.show()

def plot_stacked(ts, labels, title="Time series par région"):
    T = ts.shape[1]
    t = np.arange(T)
    n = ts.shape[0]

    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(10, 7))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t, ts[i], linewidth=1)
        ax.set_ylabel(str(int(labels[i])), rotation=0, labelpad=15)
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (frame)")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_10_ts(timeseries, n=10, title="10 random pixels time series"):
    """Affiche n time series empilées, avec une hauteur totale fixe.
    timeseries: (N, T)
    n: nombre de séries à afficher (choisies aléatoirement si N > n)
    """


    n = min(n, timeseries.shape[0])
    T = timeseries.shape[1]
    t = np.arange(T)

    # Hauteur totale fixe (8 pouces max)
    total_height = 8
    fig_height = total_height
    fig_width = 10

    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        sharex=True,
        figsize=(fig_width, fig_height)
    )

    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        a = np.random.randint(1,high=len(timeseries))
        ax.plot(t, timeseries[i+a], linewidth=1)
        ax.set_ylabel(f"région {i+a}", rotation=0, labelpad=50)
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (frame)")
    fig.suptitle(title)

    plt.tight_layout()
    plt.show()


def plot_10_ts_with_mask_clean(
    timeseries,
    sub_mask,
    n=10,
    subgroup_ids=None,
    title="10 sous-groupes aléatoires",
    seed=None,
    cmap_mask="tab20",
    annotate_selected_only=True,
    fontsize_ids=7
):
    """
    Affiche un masque de sous-groupes à gauche et n time series à droite,
    avec un layout plus propre.

    Hypothèse:
    - timeseries[k] correspond au sous-groupe k
    - sub_mask contient ces mêmes IDs
    - fond possible: NaN ou valeur négative

    Args:
    timeseries: array (n_subgroups, T)
    sub_mask: array (H, W) avec les mêmes IDs que timeseries
    n: nombre de sous-groupes à afficher (choisis aléatoirement si subgroup_ids=None)
    subgroup_ids: liste d'IDs de sous-groupes à afficher (si None, choix aléatoire parmi les IDs présents dans sub_mask)
    title: titre de la figure
    seed: graine pour la reproductibilité du choix aléatoire
    cmap_mask: colormap pour afficher le masque à gauche
    annotate_selected_only: si True, n'affiche que les IDs des sous-groupes sélectionnés sur le masque; sinon, affiche tous les IDs présents
    fontsize_ids: taille de la police pour les IDs affichés sur le masque
    """
    rng = np.random.default_rng(seed)

    N, T = timeseries.shape
    t = np.arange(T)

    # Choix des sous-groupes
    if subgroup_ids is None:
        valid_ids = np.unique(sub_mask[np.isfinite(sub_mask)])
        valid_ids = valid_ids[valid_ids >= 0].astype(int)
        n = min(n, len(valid_ids))
        chosen_ids = rng.choice(valid_ids, size=n, replace=False)
    else:
        chosen_ids = np.array(subgroup_ids, dtype=int)
        n = len(chosen_ids)

    # Figure
    fig = plt.figure(figsize=(13, max(7, 0.8 * n)))
    gs = gridspec.GridSpec(
        nrows=n,
        ncols=2,
        width_ratios=[1.4, 3.5],
        wspace=0.5,
        hspace=0.1
    )

    # -------------------------
    # Masque à gauche
    # -------------------------
    ax_mask = fig.add_subplot(gs[:, 0])

    mask_plot = sub_mask.astype(float).copy()

    # gérer fond négatif si besoin
    if np.nanmin(mask_plot) < 0:
        mask_plot[mask_plot < 0] = np.nan

    ax_mask.imshow(mask_plot, cmap=cmap_mask, interpolation="nearest")

    # overlay pour mettre en évidence les groupes choisis
    overlay = np.full(sub_mask.shape, np.nan, dtype=float)
    overlay[np.isin(sub_mask, chosen_ids)] = 1.0
    ax_mask.imshow(overlay, cmap="autumn", alpha=0.45, interpolation="nearest")

    # Écrire seulement les IDs sélectionnés
    ids_to_annotate = chosen_ids if annotate_selected_only else np.unique(mask_plot[np.isfinite(mask_plot)]).astype(int)

    for sg_id in ids_to_annotate:
        coords = np.argwhere(sub_mask == sg_id)
        if len(coords) == 0:
            continue

        r_mean, c_mean = coords.mean(axis=0)

        ax_mask.text(
            c_mean, r_mean, f"{sg_id}",
            ha="center",
            va="center",
            fontsize=fontsize_ids,
            color="white",
            fontweight="bold",
            bbox=dict(
                facecolor="black",
                alpha=0.65,
                edgecolor="white",
                boxstyle="round,pad=0.18"
            )
        )

    ax_mask.set_title("Sous-groupes sélectionnés", fontsize=14, pad=10)
    ax_mask.set_xticks([])
    ax_mask.set_yticks([])
    ax_mask.set_xlim(-0.5, sub_mask.shape[1]-0.5)
    ax_mask.set_ylim(sub_mask.shape[0]-0.5, -0.5)

    # -------------------------
    # Courbes à droite
    # -------------------------
    ts_axes = []

    y_global_min = np.nanmin(timeseries[chosen_ids])
    y_global_max = np.nanmax(timeseries[chosen_ids])

    for i, sg_id in enumerate(chosen_ids):
        ax = fig.add_subplot(gs[i, 1], sharex=ts_axes[0] if ts_axes else None)
        ts_axes.append(ax)

        ax.plot(t, timeseries[sg_id], linewidth=1.2)

        # petit label discret
        ax.text(
            -0.06, 0.5, f"région {sg_id}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold"
        )

        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax_mask.axis('off')
        ax.set_ylim(y_global_min, y_global_max)

        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)

    ts_axes[-1].set_xlabel("Time (frame)", fontsize=12)

    fig.suptitle(title, fontsize=16, y=0.98)
    plt.show()


def subdivide_mask_by_spatial_clustering(region_mask,
                                         target_size=100,
                                         background_value=np.nan,
                                         random_state=0,
                                         n_init=10):
    """
    Subdivise chaque région d'un masque en sous-groupes spatialement proches,
    sans imposer une connectivité stricte. Adapté aux régions trouées par des
    pixels morts.

    Paramètres
    ----------
    region_mask : ndarray (H, W)
        Masque 2D des régions.
        Les labels de régions sont numériques.
        Le fond peut être NaN ou une valeur explicite.
    target_size : int
        Taille visée des sous-groupes.
    background_value :
        Valeur de fond à utiliser dans le masque de sortie.
    random_state : int
        Graine pour rendre le clustering reproductible.
    n_init : int
        Nombre d'initialisations pour KMeans.

    Retour
    ------
    subgroup_mask : ndarray (H, W)
        Masque 2D avec un label unique par sous-groupe.
        Le fond vaut NaN si background_value=np.nan, sinon background_value.
    subgroup_info : dict
        Dictionnaire :
            subgroup_id -> {
                "parent_region": label régional d'origine,
                "local_subgroup_id": index local dans la région,
                "n_pixels": nombre de pixels du sous-groupe,
                "centroid": (row_mean, col_mean)
            }
    """

    region_mask = np.asarray(region_mask)
    H, W = region_mask.shape

    # Préparation du masque de sortie
    if np.issubdtype(type(background_value), np.floating) and np.isnan(background_value):
        subgroup_mask = np.full((H, W), np.nan, dtype=float)
    else:
        subgroup_mask = np.full((H, W), background_value, dtype=region_mask.dtype)

    # Trouver les labels régionaux valides
    if np.issubdtype(region_mask.dtype, np.floating) and np.isnan(background_value):
        valid = np.isfinite(region_mask)
        region_labels = np.unique(region_mask[valid])
    else:
        valid = region_mask != background_value
        region_labels = np.unique(region_mask[valid])

    subgroup_info = {}
    global_subgroup_id = 0

    for region_label in region_labels:
        coords = np.argwhere(region_mask == region_label)  # shape (N, 2)
        N = len(coords)

        if N == 0:
            continue

        # Nombre de sous-groupes voulu pour cette région
        n_clusters = int(np.ceil(N / target_size))
        n_clusters = max(1, min(n_clusters, N))

        # Cas trivial : région plus petite que target_size
        if n_clusters == 1:
            subgroup_mask[coords[:, 0], coords[:, 1]] = global_subgroup_id

            centroid = coords.mean(axis=0)
            subgroup_info[global_subgroup_id] = {
                "parent_region": region_label,
                "local_subgroup_id": 0,
                "n_pixels": N,
                "centroid": (float(centroid[0]), float(centroid[1]))
            }
            global_subgroup_id += 1
            continue

        # Clustering spatial
        km = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init
        )
        local_labels = km.fit_predict(coords)

        # Réindexation locale pour avoir 0,1,2,... dans l'ordre
        unique_local = np.unique(local_labels)

        for local_subgroup_id, loc_lab in enumerate(unique_local):
            chunk = coords[local_labels == loc_lab]
            subgroup_mask[chunk[:, 0], chunk[:, 1]] = global_subgroup_id

            centroid = chunk.mean(axis=0)
            subgroup_info[global_subgroup_id] = {
                "parent_region": region_label,
                "local_subgroup_id": local_subgroup_id,
                "n_pixels": len(chunk),
                "centroid": (float(centroid[0]), float(centroid[1]))
            }

            global_subgroup_id += 1

     # Trier les ids selon leur centroïde
    sorted_ids = sorted(
        subgroup_info.keys(),
        key=lambda i: (
            subgroup_info[i]["centroid"][0],   # row (haut → bas)
            -subgroup_info[i]["centroid"][1]   # col (droite → gauche)
        )
    )
    # ------------------------------------------------------------------
    # Réordonnancement global des sous-groupes (haut-droite → bas-gauche)
    # ------------------------------------------------------------------
    
    # Mapping ancien id → nouvel id
    old_to_new = {old: new for new, old in enumerate(sorted_ids)}

    # Appliquer au masque
    new_mask = subgroup_mask.copy()
    for old, new in old_to_new.items():
        new_mask[subgroup_mask == old] = new

    subgroup_mask = new_mask

    # Mettre à jour subgroup_info
    new_info = {}
    for old, new in old_to_new.items():
        new_info[new] = subgroup_info[old]

    subgroup_info = new_info

    return subgroup_mask, subgroup_info

def clean_region_mask(region_mask, min_component_size=50, background_value=np.nan):


    """
    Nettoie le masque de régions en réassignant les petites composantes connexes
    isolées à la région voisine la plus proche (par dilatation successive).

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
    cleaned[to_fix] = np.nan  # marquer temporairement comme fond
    remaining = to_fix.copy()

    while remaining.any():
        # Dilater le masque des pixels valides d'un pixel (4-connexe)
        valid_now = np.isfinite(cleaned)
        dilated = ndimage.binary_dilation(valid_now, structure=ndimage.generate_binary_structure(2, 1))
        newly_covered = dilated & remaining

        # Assigner à chaque pixel nouvellement couvert le label de son voisin valide
        for r, c in np.argwhere(newly_covered):
            neighbors = [cleaned[r+dr, c+dc]
                         for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                         if 0 <= r+dr < cleaned.shape[0] and 0 <= c+dc < cleaned.shape[1]
                         and np.isfinite(cleaned[r+dr, c+dc])]
            if neighbors:
                cleaned[r, c] = max(set(neighbors), key=neighbors.count)
                remaining[r, c] = False

    return cleaned