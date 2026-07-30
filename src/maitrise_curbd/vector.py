import pickle
import numpy as np
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

### Loader les pkl en array de courants CURBD

def load_curbd_currents_array(pkl_path):
    """
    Charge un fichier CURBD .pkl et retourne un array de taille
    (n_regions², T).

    Chaque ligne correspond à un courant CURBD (source → cible).
    L'ordre est :
        (0,0), (0,1), ..., (0,5),
        (1,0), ..., (5,5)
    """

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    currents = data["currents_curves"]
    n_regions = len(data["regions"])

    X = np.stack(
        [
            currents[(iTarget, iSource)]
            for iTarget in range(n_regions)
            for iSource in range(n_regions)
        ],
        axis=0,
    )

    return X

### Diminuer les courants CURBD à 5 métriques par courant

def compute_curbd_metrics(currents_array, dt=1.0):
    """
    Calcule 5 métriques pour chaque courant CURBD.

    Paramètres
    ----------
    currents_array : np.ndarray, shape (n_currents, T)
        Chaque ligne représente un courant CURBD.

    dt : float, default=1.0
        Intervalle temporel entre deux points, en secondes.
        Par exemple :
            dt = 0.0833 pour des données à 12 Hz.

    Retour
    ------
    metrics : np.ndarray, shape (n_currents, 5)
        Colonnes :
            0 : RMS
            1 : P95 - P5
            2 : aire sous la courbe signée
            3 : skewness
            4 : kurtosis
    """

    currents_array = np.asarray(currents_array, dtype=float)

    if currents_array.ndim != 2:
        raise ValueError(
            "currents_array doit être un array 2D de forme (n_currents, T)."
        )

    n_currents = currents_array.shape[0]
    metrics = np.full((n_currents, 5), np.nan)

    for i in range(n_currents):

        current = currents_array[i]

        # Retirer les valeurs NaN
        valid = np.isfinite(current)
        current_valid = current[valid]

        if current_valid.size == 0:
            continue

        # 1. RMS
        rms = np.sqrt(np.mean(current_valid**2))

        # 2. Étendue robuste
        p95_p5 = (
            np.percentile(current_valid, 95)
            - np.percentile(current_valid, 5)
        )

        # 3. Aire sous la courbe signée
        auc = np.trapezoid(current_valid, dx=dt)

        # 4. Skewness
        skewness = skew(
            current_valid,
            bias=False,
            nan_policy="omit"
        )

        # 5. Kurtosis excédentaire
        kurt = kurtosis(
            current_valid,
            fisher=True,
            bias=False,
            nan_policy="omit"
        )

        metrics[i] = [
            rms,
            p95_p5,
            auc,
            skewness,
            kurt
        ]

    return metrics

import numpy as np
import matplotlib.pyplot as plt

def plot_curbd_metrics(metrics):
    """
    Affiche les métriques des courants CURBD sous forme de tableau coloré
    avec les valeurs numériques dans chaque case.

    metrics : array (36, 5)
    """

    metrics = np.asarray(metrics)

    metric_names = ["RMS", "P95-P5", "AUC", "Skew", "Kurt"]

    n_currents = metrics.shape[0]
    current_labels = [
        f"S{i%6}→C{i//6}"
        for i in range(n_currents)
        ]

    # Normalisation UNIQUEMENT pour les couleurs
    colors = metrics.copy()
    mn = np.nanmin(colors, axis=0)
    mx = np.nanmax(colors, axis=0)
    colors = (colors - mn) / (mx - mn + 1e-12)

    fig, ax = plt.subplots(figsize=(7, 11))

    im = ax.imshow(colors, cmap="viridis", aspect="auto")

    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_xticklabels(metric_names, fontsize=11)

    ax.set_yticks(np.arange(n_currents))
    ax.set_yticklabels(current_labels, fontsize=8)

    # Valeurs numériques
    for i in range(n_currents):
        for j in range(len(metric_names)):
            value = metrics[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if colors[i, j] < 0.5 else "black",
                fontsize=7,
            )

    ax.set_xlabel("Métriques")
    ax.set_ylabel("Courants CURBD")
    ax.set_title("Métriques des courants CURBD")

    plt.tight_layout()
    plt.show()