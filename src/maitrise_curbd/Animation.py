from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_subregions(
    ts,
    masque_sub,
    interval=50,
    labels=None,
    percentile_min=1,
    percentile_max=99,
    cmap_name="RdBu_r",
    center_zero=True,
):
    """
    Anime l'activité temporelle des sous-régions après prétraitement,
    notamment après régression du signal global.

    Paramètres
    ----------
    ts : array
        Séries temporelles des sous-régions.

        Formats acceptés :
        - (n_regions, T)
        - (T, n_regions)

        La fonction tente de détecter automatiquement l'orientation.

    masque_sub : array (H, W)
        Masque spatial contenant un label par sous-région.
        Le fond peut être représenté par NaN.

    interval : int
        Intervalle entre les frames, en millisecondes.

    labels : array-like, optionnel
        Labels du masque correspondant aux lignes de ts.

        Si None, les labels sont récupérés avec :
        np.unique(masque_sub[~np.isnan(masque_sub)])

        Ils sont triés en ordre croissant.

    percentile_min, percentile_max : float
        Percentiles utilisés pour fixer l'échelle de couleurs.

    cmap_name : str
        Colormap Matplotlib.

    center_zero : bool
        Si True, impose une échelle symétrique autour de zéro.
        Recommandé après régression du signal global.

    Retours
    -------
    ani : matplotlib.animation.FuncAnimation
        Objet animation. Il faut le conserver dans une variable.
    """

    ts = np.asarray(ts, dtype=np.float32)
    masque_sub = np.asarray(masque_sub, dtype=np.float32)

    if ts.ndim != 2:
        raise ValueError(
            f"ts doit être un tableau 2D, mais sa forme est {ts.shape}."
        )

    if masque_sub.ndim != 2:
        raise ValueError(
            f"masque_sub doit être un tableau 2D, "
            f"mais sa forme est {masque_sub.shape}."
        )

    # ------------------------------------------------------------
    # Labels réellement présents dans le masque
    # ------------------------------------------------------------
    labels_masque = np.unique(
        masque_sub[np.isfinite(masque_sub)]
    )

    if labels is None:
        labels = labels_masque
    else:
        labels = np.asarray(labels)

    n_regions = len(labels)

    # ------------------------------------------------------------
    # Détection de l'orientation de ts
    # ------------------------------------------------------------
    if ts.shape[0] == n_regions:
        # ts : régions × temps
        ts_regions = ts

    elif ts.shape[1] == n_regions:
        # ts : temps × régions
        ts_regions = ts.T

    else:
        raise ValueError(
            "Impossible d'associer ts aux régions du masque.\n"
            f"Nombre de labels : {n_regions}\n"
            f"Forme de ts : {ts.shape}\n\n"
            "Il faut que l'une des dimensions de ts soit égale "
            "au nombre de sous-régions."
        )

    n_regions_ts, T = ts_regions.shape

    # ------------------------------------------------------------
    # Vérification labels ↔ séries temporelles
    # ------------------------------------------------------------
    labels_absents = [
        label
        for label in labels
        if not np.any(masque_sub == label)
    ]

    if labels_absents:
        raise ValueError(
            "Certains labels fournis ne sont pas présents dans le masque : "
            f"{labels_absents}"
        )

    # ------------------------------------------------------------
    # Construction d'une table label -> index dans ts
    #
    # On crée directement un masque contenant les indices des lignes de ts.
    # Cela évite de refaire les comparaisons masque_sub == label à chaque frame.
    # ------------------------------------------------------------
    index_map = np.full(
        masque_sub.shape,
        -1,
        dtype=np.int32
    )

    for region_index, label_value in enumerate(labels):
        index_map[masque_sub == label_value] = region_index

    pixels_valides = index_map >= 0

    # ------------------------------------------------------------
    # Échelle de couleurs
    # ------------------------------------------------------------
    valeurs_valides = ts_regions[np.isfinite(ts_regions)]

    if valeurs_valides.size == 0:
        raise ValueError("ts ne contient aucune valeur finie.")

    pmin = np.nanpercentile(
        valeurs_valides,
        percentile_min
    )

    pmax = np.nanpercentile(
        valeurs_valides,
        percentile_max
    )

    if center_zero:
        amplitude = max(abs(pmin), abs(pmax))

        if amplitude == 0:
            amplitude = 1.0

        vmin = -amplitude
        vmax = amplitude

    else:
        vmin = pmin
        vmax = pmax

        if vmin == vmax:
            vmax = vmin + 1.0

    # ------------------------------------------------------------
    # Fonction de reconstruction d'une frame spatiale
    # ------------------------------------------------------------
    def create_frame(frame_index):

        frame_image = np.full(
            masque_sub.shape,
            np.nan,
            dtype=np.float32
        )

        frame_image[pixels_valides] = (
            ts_regions[
                index_map[pixels_valides],
                frame_index
            ]
        )

        return frame_image

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("black")

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        create_frame(0),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    colorbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Activité de la sous-région\n"
        "(signal global régressé)"
    )

    # Utiliser un texte dans l'axe évite l'empilement des titres
    frame_text = ax.text(
        0.5,
        1.02,
        f"Frame 0/{T - 1} — signal global régressé",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
    )

    ax.axis("off")

    # ------------------------------------------------------------
    # Mise à jour
    # ------------------------------------------------------------
    def update(frame_index):

        im.set_data(
            create_frame(frame_index)
        )

        frame_text.set_text(
            f"Frame {frame_index}/{T - 1} "
            "— signal global régressé"
        )

        return im, frame_text

    # blit=False est plus fiable pour les textes et selon les backends
    ani = FuncAnimation(
        fig,
        update,
        frames=T,
        interval=interval,
        blit=False,
        repeat=True,
    )

    plt.tight_layout()
    plt.show()

    return ani

def animate_dff(X, roi_mask, interval=50):

    X = np.asarray(X, dtype=np.float32)
    roi_mask = np.asarray(roi_mask, dtype=bool)

    if X.ndim != 3:
        raise ValueError(
            f"X doit avoir la forme (T, H, W), mais a la forme {X.shape}."
        )

    if roi_mask.shape != X.shape[1:]:
        raise ValueError(
            f"roi_mask {roi_mask.shape} incompatible avec X {X.shape}"
        )

    X = np.where(
        roi_mask[None, :, :],
        X,
        np.nan
    )

    F0 = np.nanmedian(
        X,
        axis=0,
        keepdims=True
    )

    F0[
        (F0 == 0) |
        ~np.isfinite(F0)
    ] = np.nan

    X_dff = (X - F0) / F0

    T = X_dff.shape[0]

    cmap = plt.cm.Greens.copy()
    cmap.set_bad("black")

    vmin = np.nanpercentile(
        X_dff,
        1
    )

    vmax = np.nanpercentile(
        X_dff,
        99
    )

    fig, ax = plt.subplots()

    im = ax.imshow(
        X_dff[0],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Texte dans l'axe plutôt qu'un titre Matplotlib
    frame_text = ax.text(
        0.5,
        1.02,
        "Frame 0 — ΔF/F",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
    )

    ax.axis("off")

    def update(frame):

        im.set_data(
            X_dff[frame]
        )

        frame_text.set_text(
            f"Frame {frame}/{T - 1} — ΔF/F"
        )

        return im, frame_text

    ani = FuncAnimation(
        fig,
        update,
        frames=T,
        interval=interval,
        blit=False,
    )

    plt.tight_layout()
    plt.show()

    return ani
