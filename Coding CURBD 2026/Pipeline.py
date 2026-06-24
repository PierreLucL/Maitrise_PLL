#### Importations et fonctions pour l'analyse des données GCaMP ####

import h5py
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
from scipy import ndimage
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import percentile_filter
import pickle
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgb

#### LOADER LE FICHIER H5 ####

def load_gcamp(h5_path):
    """
    Ouvre le fichier H5 et retourne le dataset GCaMP en numpy array.

    Args:
        h5_path: chemin vers le fichier .h5

    Retourne:
        data_gcamp: array (T, H, W) float32, (1440, 238, 261)
        mask_registration: array (H, W) int32, (238, 261)
    """
    with h5py.File(h5_path, "r") as f:
        data_gcamp = f["data/3d/GCaMP"][:]
        mask_registration = f["registration/atlas"][:]
    return data_gcamp, mask_registration

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

### EN EXTRAIRE LES TIMESERIES ###

def extract_timeseries_du_tenseur(X, mask):
    """
    X: (T,H,W) float (GCaMP) avec NaN possibles
    mask: (H,W) labels avec NaN = fond
    Retourne:
      ts: (n_labels, T) moyenne par label en ignorant NaN de X
    """
    T, H, W = X.shape

    #Détection des pixels valides
    valid = ~np.isnan(mask)

    # labels int; fond -> -1
    m = np.full((H, W), -1, dtype=np.int32)
    m[valid] = np.rint(mask[valid]).astype(np.int32)

    labels_flat = m.reshape(-1)
    valid_flat = labels_flat != -1
    labels_valid = labels_flat[valid_flat]

    labels = np.unique(labels_valid)
    lab2i = {lab: i for i, lab in enumerate(labels)}
    idx = np.array([lab2i[lab] for lab in labels_valid], dtype=np.int32)

    X_flat = X.reshape(T, H * W)
    X_valid = X_flat[:, valid_flat]  # (T, Nvalid)

    L = len(labels)
    ts = np.full((L, T), np.nan, dtype=np.float64)

    for t in range(T):
        w = X_valid[t].astype(np.float64)
        ok = np.isfinite(w)
        if not np.any(ok):
            continue

        sums = np.bincount(idx[ok], weights=w[ok], minlength=L)
        den  = np.bincount(idx[ok], minlength=L).astype(np.float64)
        ts[:, t] = sums / np.where(den == 0, np.nan, den)
    return ts

### REGRESS OUT ###

def regress_out_global_signal(ts, return_global=False):
    """
    Régress-out le signal moyen de chaque time series.

    Paramètres
    ----------
    ts : ndarray
        Matrice de forme (N, T), avec N régions et T temps.

    return_global : bool
        Si True, retourne aussi le signal global utilisé.

    Retour
    ------
    ts_resid : ndarray
        Matrice (N, T) après retrait de la composante expliquée
        linéairement par le signal global.
    """

    ts = np.asarray(ts, dtype=float)

    if ts.ndim != 2:
        raise ValueError("ts doit être de forme (N, T).")

    # signal global moyen à chaque temps
    global_signal = np.nanmean(ts, axis=0)

    # centrage
    g = global_signal - np.nanmean(global_signal)

    ts_resid = np.empty_like(ts)

    denom = np.nansum(g**2)

    if denom == 0:
        raise ValueError("Le signal global est constant; régression impossible.")

    for i in range(ts.shape[0]):
        y = ts[i]
        y_mean = np.nanmean(y)
        y_centered = y - y_mean

        beta = np.nansum(y_centered * g) / denom

        # résidu = signal original - composante prédite par le signal global
        ts_resid[i] = y_centered - beta * g

    return (ts_resid, global_signal) if return_global else ts_resid

### Rolling window

def compute_dff(ts, fs=3.0, window_sec=60, percentile=8, eps=1e-8):
    """
    Calcule ΔF/F avec baseline glissante basée sur un percentile.

    Paramètres
    ----------
    ts : ndarray (N, T)
        Fluorescence brute.
    fs : float
        Fréquence d'acquisition (Hz).
    window_sec : float
        Taille de la fenêtre glissante (secondes).
    percentile : float
        Percentile utilisé pour F0.
    eps : float
        Évite les divisions par zéro.

    Retour
    -------
    dff : ndarray (N, T)
        Signal ΔF/F.
    F0 : ndarray (N, T)
        Baseline estimée.
    """

    window_frames = int(round(window_sec * fs))

    F0 = percentile_filter(
        ts,
        percentile=percentile,
        size=(1, window_frames),
        mode='reflect'
    )

    dff = (ts - F0) / (F0 + eps)

    return dff

#### LISSAGE ####

def smooth_timeseries(ts, sigma=2, window=None):
    """
    Lisse des séries temporelles avec un filtre gaussien.

    Paramètres
    ----------
    ts : np.ndarray
        Matrice de séries temporelles de forme (N, T),
        où N = nombre de régions et T = nombre de pas de temps.
    sigma : float
        Écart-type du noyau gaussien, en nombre de pas de temps.
    window : int ou None
        Taille de la fenêtre temporelle utilisée pour tronquer le noyau.
        Si None, scipy utilise truncate=4 par défaut.

    Retour
    ------
    ts_smooth : np.ndarray
        Matrice lissée de même forme que ts.
    """

    ts = np.asarray(ts, dtype=float)

    if ts.ndim != 2:
        raise ValueError("ts doit être une matrice 2D de forme (N, T).")

    if sigma <= 0:
        return ts.copy()

    if window is None:
        return gaussian_filter1d(ts, sigma=sigma, axis=1, mode="nearest")

    if window < 1:
        raise ValueError("window doit être >= 1.")

    truncate = (window / 2) / sigma

    ts_smooth = gaussian_filter1d(
        ts,
        sigma=sigma,
        axis=1,
        mode="nearest",
        truncate=truncate
    )

    return ts_smooth

### PLOT IT ###

def cmap_masque(masque):
    base = plt.cm.tab20.colors  # 20 couleurs fixes
    n = int(np.nanmax(masque)) + 1  # nombre max de labels
    colors = [base[i % 20] for i in range(n)]
    cmap_masque = ListedColormap(colors)
    return cmap_masque

def plot_10_ts_with_mask_and_similarity(
    timeseries,
    sub_mask,
    n_pixels,
    souris,
    n=10,
    subgroup_ids=None,
    seed=None,
    annotate_selected_only=True,
    fontsize_ids=10,
):
    """
    Affiche :
    - à gauche haut : masque des sous-groupes avec groupes choisis surlignés
    - à gauche bas  : matrice de similarité de Pearson entre les groupes choisis
    - à droite      : time series des groupes choisis

    Hypothèses :
    - timeseries[k] correspond au sous-groupe k
    - sub_mask contient ces mêmes IDs
    - fond de sub_mask = NaN ou valeur négative
    """

    rng = np.random.default_rng(seed)

    timeseries = np.asarray(timeseries)
    sub_mask = np.asarray(sub_mask)

    N, T = timeseries.shape
    t = np.arange(T)

    # Choix des sous-groupes
    if subgroup_ids is None:
        valid_ids = np.unique(sub_mask[np.isfinite(sub_mask)])
        valid_ids = valid_ids[valid_ids >= 0].astype(int)
        n = min(n, len(valid_ids))
        chosen_ids = np.sort(rng.choice(valid_ids, size=n, replace=False))
    else:
        chosen_ids = np.sort(np.array(subgroup_ids, dtype=int))
        n = len(chosen_ids)

    # Sous-ensemble de TS
    ts_sel = timeseries[chosen_ids]

    # Matrice de similarité
    sim = np.corrcoef(ts_sel)
    sim_tot = np.corrcoef(timeseries)
    # Figure
    fig = plt.figure(figsize=(16, max(8, 0.8 * n)))
    outer = gridspec.GridSpec(
        nrows=1,
        ncols=2,
        width_ratios=[1.35, 3.65],
        wspace=0.4
    )

    # Colonne de gauche découpée en 2
    left = gridspec.GridSpecFromSubplotSpec(
        nrows=2,
        ncols=1,
        subplot_spec=outer[0],
        height_ratios=[1.15, 1.0],
        hspace=0.5
    )

    # -------------------------
    # Haut gauche : masque
    # -------------------------
    ax_mask = fig.add_subplot(left[0])

    mask_plot = sub_mask.astype(float).copy()
    if np.nanmin(mask_plot) < 0:
        mask_plot[mask_plot < 0] = np.nan

    ax_mask.imshow(mask_plot, cmap=cmap_masque(sub_mask), interpolation="nearest")

    # Overlay pour mettre en évidence les groupes choisis
    overlay = np.full(sub_mask.shape, np.nan, dtype=float)
    overlay[np.isin(sub_mask, chosen_ids)] = 1.0
    ax_mask.imshow(overlay, cmap="autumn", alpha=0.45, interpolation="nearest")

    ids_to_annotate = (
        chosen_ids
        if annotate_selected_only
        else np.unique(mask_plot[np.isfinite(mask_plot)]).astype(int)
    )

    for sg_id in ids_to_annotate:
        coords = np.argwhere(sub_mask == sg_id)
        if len(coords) == 0:
            continue

        r_mean, c_mean = coords.mean(axis=0)

        ax_mask.text(
            c_mean,
            r_mean,
            f"{sg_id}",
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

    ax_mask.set_title("Sous-groupes sélectionnés", fontsize=13, pad=8)
    ax_mask.set_aspect("equal")
    ax_mask.axis("off")
    ax_mask.set_xlim(-0.5, sub_mask.shape[1] - 0.5)
    ax_mask.set_ylim(sub_mask.shape[0] - 0.5, -0.5)

    # -------------------------
    # Bas gauche : similarité
    # -------------------------
    ax_sim = fig.add_subplot(left[1])

    im = ax_sim.imshow(sim, vmin=-1, vmax=1, interpolation="nearest")
    ax_sim.set_title(f"Similarité de Pearson (Moy globale :{np.mean(sim_tot):.2f})", fontsize=13, pad=8)

    ax_sim.set_xticks(np.arange(n))
    ax_sim.set_yticks(np.arange(n))
    ax_sim.set_xticklabels(chosen_ids, rotation=90)
    ax_sim.set_yticklabels(chosen_ids)

    # petite grille visuelle
    ax_sim.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax_sim.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax_sim.grid(which="minor", alpha=0.2)
    ax_sim.tick_params(which="minor", bottom=False, left=False)

    # afficher les coefficients dans les cases
    for i in range(n):
        for j in range(n):
            val = sim[i, j]
            if np.isfinite(val):
                ax_sim.text(
                    j, i, f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black"
                )

    cbar = fig.colorbar(im, ax=ax_sim, fraction=0.046, pad=0.04)
    cbar.set_label("r de Pearson")

    # -------------------------
    # Droite : time series
    # -------------------------
    right = gridspec.GridSpecFromSubplotSpec(
        nrows=n,
        ncols=1,
        subplot_spec=outer[1],
        hspace=0.22
    )

    ts_axes = []

    y_global_min = np.nanmin(ts_sel)
    y_global_max = np.nanmax(ts_sel)

    for i, sg_id in enumerate(chosen_ids):
        ax = fig.add_subplot(right[i], sharex=ts_axes[0] if ts_axes else None)
        ts_axes.append(ax)

        ax.plot(t, timeseries[sg_id], linewidth=1.2)

        ax.text(
            -0.055, 0.5, f"rég. {sg_id}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold"
        )

        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(y_global_min, y_global_max)

        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)

    ts_axes[-1].set_xlabel("Time (frame)", fontsize=12)
    ts_axes[0].set_title(r"$F-\Delta F_{global}$", fontsize=13, pad=8)
    fig.suptitle(f"10 sous-groupes aléatoires de {n_pixels} pixels, souris {souris}", fontsize=16, y=0.995)
    plt.show()

def plot_region_highlight(masque_sub, region_indices, title="Région sélectionnée"):
    """
    masque_sub : array 2D contenant les labels des sous-régions
    region_indices : liste ou array des labels à highlight
    """

    region_indices = np.array(region_indices)

    # masque booléen : True là où le pixel appartient à la région
    highlight = np.isin(masque_sub, region_indices)

    plt.figure(figsize=(8, 8))

    # fond : toutes les sous-régions en gris
    plt.imshow(masque_sub, cmap="gray", alpha=0.35)

    # overlay : seulement la région sélectionnée
    overlay = np.where(highlight, 1, np.nan)
    plt.imshow(overlay, cmap="autumn", alpha=0.8)

    plt.title(title)
    plt.axis("off")
    plt.show()

def gradient_line(
    x,
    y,
    ax,
    color_start,
    color_end,
    lw=2
):

    cmap = LinearSegmentedColormap.from_list(
        "source_target",
        [color_start, color_end]
    )

    points = np.array([x, y]).T.reshape(-1, 1, 2)

    segments = np.concatenate(
        [points[:-1], points[1:]],
        axis=1
    )

    lc = LineCollection(
        segments,
        cmap=cmap,
        linewidth=lw
    )
    transition = np.linspace(-6, 6, len(segments))

    colors = 1/(1+np.exp(-transition))
    lc.set_array(colors)

    ax.add_collection(lc)

    return lc

#### CURBDING

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

### Pkl retrieve ?

def plot_curbd_currents_from_pkl(pkl_path, gradient_line):
    """
    Plot les 36 courbes de courant CURBD directement depuis un fichier .pkl.
    """

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    currents = data["currents_curves"]
    regions = data["regions"]
    masque_sub = data["masque_sub"]
    tRNN = data["tRNN"]

    n_regions = len(regions)

    region_colors = {
        0: "#0047AB",
        1: "#FF7F00",
        2: "#00A550",
        3: "#A020F0",
        4: "#E60026",
        5: "#00B7EB",
    }

    all_currents = np.concatenate(list(currents.values()))
    max_abs = np.percentile(np.abs(all_currents), 99)

    mask_rgb = np.ones((*masque_sub.shape, 3))

    for iRegion in range(n_regions):
        subregion_indices = regions[iRegion, 1]
        color = to_rgb(region_colors[iRegion])

        for idx in subregion_indices:
            mask_rgb[masque_sub == idx] = color

    if np.any(np.isnan(masque_sub)):
        mask_rgb[np.isnan(masque_sub)] = [1, 1, 1]

    fig = plt.figure(figsize=(12, 8))

    outer = GridSpec(
        1, 2,
        width_ratios=[1, 5],
        wspace=0.15,
        figure=fig
    )

    ax_mask = fig.add_subplot(outer[0, 0])
    ax_mask.imshow(mask_rgb)
    ax_mask.set_title("Régions", fontsize=10)
    ax_mask.axis("off")

    right = outer[0, 1].subgridspec(
        n_regions,
        n_regions,
        wspace=0.08,
        hspace=0.08
    )

    for iTarget in range(n_regions):
        for iSource in range(n_regions):

            ax = fig.add_subplot(right[iTarget, iSource])

            current = currents[(iTarget, iSource)]

            source_color = region_colors[iSource]
            target_color = region_colors[iTarget]

            gradient_line(
                tRNN,
                current,
                ax,
                source_color,
                target_color,
                lw=3 if iSource == iTarget else 2
            )

            ax.axhline(0, color="black", linewidth=0.4, alpha=0.25)

            ax.set_xlim(tRNN[0], tRNN[-1])
            ax.set_ylim(-max_abs, max_abs)

            if iTarget == 0:
                ax.set_title(
                    regions[iSource, 0],
                    fontsize=8,
                    color=source_color
                )

            if iSource == 0:
                ax.set_ylabel(
                    regions[iTarget, 0],
                    fontsize=8,
                    color=target_color
                )

            if iTarget != n_regions - 1:
                ax.set_xticklabels([])

            if iSource != 0:
                ax.set_yticklabels([])

            ax.tick_params(axis="both", labelsize=6, length=2)

            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_alpha(0.5)

    fig.suptitle(
        "Courants CURBD source → cible",
        fontsize=14,
        y=0.98
    )

    plt.show()


### Synthétiques ?

def generate_synthetic_gcamp_rnn(
    height=40,
    width=60,
    T=1000,
    g=1.5,
    dt=0.33,
    tau=1.0,
    noise_std=0.02,
    same_region_strength=2.5,
    spatial_decay=8.0,
    calcium_tau=1.2,
    apply_gcamp=True,
    seed=None,
):
    """
    Génère des données synthétiques type GCaMP avec 6 régions carrées.

    Returns
    -------
    mask : array (height, width)
        Masque contenant les labels 0 à 5.
    W : array (N, N)
        Matrice de connectivité vraie entre tous les pixels.
    X : array (N, T)
        Séries temporelles simulées, N pixels x T temps.
    """

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------------
    # 1. Masque : 6 régions en grille 2 x 3
    # ---------------------------------------------------------------------
    mask = np.zeros((height, width), dtype=int)

    n_rows = 2
    n_cols = 3

    region_h = height // n_rows
    region_w = width // n_cols

    label = 0
    for i in range(n_rows):
        for j in range(n_cols):
            y0 = i * region_h
            y1 = (i + 1) * region_h if i < n_rows - 1 else height
            x0 = j * region_w
            x1 = (j + 1) * region_w if j < n_cols - 1 else width

            mask[y0:y1, x0:x1] = label
            label += 1

    N = height * width
    labels = mask.ravel()

    # Coordonnées spatiales des pixels
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    coords = np.column_stack([yy.ravel(), xx.ravel()])

    # ---------------------------------------------------------------------
    # 2. Matrice de connectivité structurée
    # ---------------------------------------------------------------------
    dy = coords[:, 0][:, None] - coords[:, 0][None, :]
    dx = coords[:, 1][:, None] - coords[:, 1][None, :]
    dist = np.sqrt(dx**2 + dy**2)

    spatial_kernel = np.exp(-dist / spatial_decay)

    same_region = labels[:, None] == labels[None, :]

    structure = spatial_kernel.copy()
    structure[same_region] *= same_region_strength

    # Poids aléatoires structurés
    W = rng.normal(0, 1, size=(N, N)).astype(np.float32)
    W *= structure.astype(np.float32)

    # Pas d'auto-connexion directe
    np.fill_diagonal(W, 0)

    # Normalisation par rayon spectral
    eigvals = np.linalg.eigvals(W)
    spectral_radius = np.max(np.abs(eigvals))

    if spectral_radius > 0:
        W = W / spectral_radius * g

    W = W.astype(np.float32)

    # ---------------------------------------------------------------------
    # 3. Simulation RNN
    # ---------------------------------------------------------------------
    H = rng.normal(0, 0.1, size=N).astype(np.float32)
    X_neural = np.zeros((N, T), dtype=np.float32)

    for t in range(T):
        R = np.tanh(H)
        X_neural[:, t] = R

        noise = rng.normal(0, noise_std, size=N).astype(np.float32)

        dH = (-H + W @ R + noise) / tau
        H = H + dt * dH

    # ---------------------------------------------------------------------
    # 4. Observation GCaMP synthétique
    # ---------------------------------------------------------------------
    if apply_gcamp:
        # filtre exponentiel approximé par un lissage gaussien temporel
        sigma_frames = calcium_tau / dt
        X = gaussian_filter1d(X_neural, sigma=sigma_frames, axis=1)

        # bruit observationnel calcium
        X += rng.normal(0, noise_std, size=X.shape).astype(np.float32)

        # normalisation par pixel
        X = X - X.mean(axis=1, keepdims=True)
        X = X / (X.std(axis=1, keepdims=True) + 1e-8)

    else:
        X = X_neural

    return mask, W, X