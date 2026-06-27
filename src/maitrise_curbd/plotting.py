import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import pickle
from matplotlib.gridspec import GridSpec

from matplotlib.colors import to_rgb

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
    lw=0.8
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

### Plot from pkl

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
                lw=0.8 if iSource == iTarget else 0.5
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
        "Courants CURBD source → cible, souris {}, n_pixels {}".format(data["row"]["souris"], data["row"]["n_pixels"]),
        fontsize=14,
        y=0.98
    )

    plt.show()

