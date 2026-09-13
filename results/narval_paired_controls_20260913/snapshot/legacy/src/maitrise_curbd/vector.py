import pickle
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

### Vectorisation des courants CURBD: on transforme les belles courbes en features comparables.

### Loader les pkl en array de courants CURBD. C'est le pont entre "objet pickle" et "matrice clean".
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
        ### On charge le run sauvegarde; ici on suppose le format avec currents_curves.
        data = pickle.load(f)

    currents = data["currents_curves"]
    n_regions = len(data["regions"])

    X = np.stack(
        [
            ### Ordre stable source/cible, sinon les features deviennent du spaghetti.
            currents[(iTarget, iSource)]
            for iTarget in range(n_regions)
            for iSource in range(n_regions)
        ],
        axis=0,
    )

    return X


### Diminuer les courants CURBD a 5 metriques par courant. Mini resume numerique, full pratique.
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

        ### On traite chaque courant seul: un courant, cinq stats, next.
        current = currents_array[i]

        ### Retirer les valeurs NaN; elles sont pas invitees aux stats.
        valid = np.isfinite(current)
        current_valid = current[valid]

        if current_valid.size == 0:
            continue

        ### 1. RMS: energie moyenne du courant.
        rms = np.sqrt(np.mean(current_valid**2))

        ### 2. Etendue robuste: amplitude sans laisser un outlier crier trop fort.
        p95_p5 = (
            np.percentile(current_valid, 95)
            - np.percentile(current_valid, 5)
        )

        ### 3. Aire signee: direction nette accumulee dans le temps.
        auc = np.trapezoid(current_valid, dx=dt)

        ### 4. Skewness: asymetrie du courant.
        skewness = skew(
            current_valid,
            bias=False,
            nan_policy="omit"
        )

        ### 5. Kurtosis: est-ce que la courbe a des gros pics dramatiques?
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


### Plot rapide des metriques pour eyeballer les courants sans ouvrir 36 figures.
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

    ### Normalisation uniquement pour les couleurs; les chiffres affiches restent les vrais.
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

    ### On ecrit les valeurs dans les cases, parce qu'une heatmap sans chiffres c'est parfois du vibes-only.
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
