#### Importations et fonctions pour l'analyse des données GCaMP ####

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import percentile_filter

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