#### Importations et fonctions pour l'analyse des données GCaMP ####

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import percentile_filter
from scipy.signal import welch

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

def compute_dff(ts, fs=12.0, window_sec=60, percentile=8, eps=1e-8):
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


def compute_mean_psd(timeseries, fs, nperseg=None):
    """
    Calcule la densité spectrale de puissance moyenne d'un ensemble
    de séries temporelles.

    Parameters
    ----------
    timeseries : np.ndarray
        Tableau de forme (N, T), avec N traces et T points temporels.
    fs : float
        Fréquence d'échantillonnage en Hz.
    nperseg : int or None
        Taille des segments pour scipy.signal.welch.
        Si None, une valeur raisonnable est choisie automatiquement.

    Returns
    -------
    freqs : np.ndarray
        Fréquences en Hz.
    mean_psd : np.ndarray
        PSD moyenne sur toutes les traces.
    psd_all : np.ndarray
        PSD individuelle de chaque trace, shape (N, F).
    """

    timeseries = np.asarray(timeseries, dtype=float)

    if timeseries.ndim != 2:
        raise ValueError(
            f"timeseries doit avoir la forme (N, T), reçu {timeseries.shape}"
        )

    n_timepoints = timeseries.shape[1]

    if nperseg is None:
        nperseg = min(1024, n_timepoints)

    # Retire la moyenne de chaque trace
    ts_centered = timeseries - np.nanmean(
        timeseries,
        axis=1,
        keepdims=True
    )

    psds = []

    for trace in ts_centered:

        if np.all(np.isnan(trace)):
            continue

        trace = np.nan_to_num(trace)

        freqs, psd = welch(
            trace,
            fs=fs,
            nperseg=nperseg,
            detrend="constant"
        )

        psds.append(psd)

    if len(psds) == 0:
        raise ValueError("Aucune trace valide pour calculer la PSD.")

    psd_all = np.asarray(psds)

    mean_psd = np.nanmean(
        psd_all,
        axis=0
    )

    return freqs, mean_psd, psd_all


def autocorrelation_1d(trace, max_lag=None):
    """
    Calcule l'autocorrélation normalisée d'une seule trace.

    Parameters
    ----------
    trace : np.ndarray
        Série temporelle 1D.
    max_lag : int or None
        Lag maximal, en nombre de frames.

    Returns
    -------
    acf : np.ndarray
        Autocorrélation normalisée, avec acf[0] = 1.
    """

    trace = np.asarray(trace, dtype=float)

    valid = np.isfinite(trace)

    if valid.sum() < 3:
        return None

    trace = trace.copy()

    # Remplissage très simple des NaN par la moyenne
    trace[~valid] = np.nanmean(trace)

    trace = trace - np.mean(trace)

    variance = np.sum(trace ** 2)

    if variance == 0:
        return None

    correlation = np.correlate(
        trace,
        trace,
        mode="full"
    )

    correlation = correlation[len(trace) - 1:]

    acf = correlation / variance

    if max_lag is not None:
        acf = acf[:max_lag + 1]

    return acf


def compute_mean_autocorrelation(
    timeseries,
    dt,
    max_lag_seconds=5.0
):
    """
    Calcule l'autocorrélation moyenne de plusieurs traces.

    Parameters
    ----------
    timeseries : np.ndarray
        Shape (N, T).
    dt : float
        Pas temporel en secondes.
    max_lag_seconds : float
        Temps maximal jusqu'auquel calculer l'autocorrélation.

    Returns
    -------
    lag_times : np.ndarray
        Lags en secondes.
    mean_acf : np.ndarray
        Autocorrélation moyenne.
    acf_all : np.ndarray
        Autocorrélations individuelles.
    """

    timeseries = np.asarray(timeseries, dtype=float)

    if timeseries.ndim != 2:
        raise ValueError(
            f"timeseries doit avoir la forme (N, T), reçu {timeseries.shape}"
        )

    max_lag = int(
        np.round(max_lag_seconds / dt)
    )

    max_lag = min(
        max_lag,
        timeseries.shape[1] - 1
    )

    acfs = []

    for trace in timeseries:

        acf = autocorrelation_1d(
            trace,
            max_lag=max_lag
        )

        if acf is not None:
            acfs.append(acf)

    if len(acfs) == 0:
        raise ValueError(
            "Aucune trace valide pour calculer l'autocorrélation."
        )

    acf_all = np.asarray(acfs)

    mean_acf = np.nanmean(
        acf_all,
        axis=0
    )

    lag_times = (
        np.arange(mean_acf.size) * dt
    )

    return lag_times, mean_acf, acf_all


def estimate_autocorrelation_timescale(
    lag_times,
    acf,
    threshold=np.exp(-1)
):
    """
    Estime le temps caractéristique comme le premier lag
    auquel l'autocorrélation passe sous 1/e.

    Une interpolation linéaire est utilisée entre les deux points.

    Parameters
    ----------
    lag_times : np.ndarray
        Lags en secondes.
    acf : np.ndarray
        Autocorrélation.
    threshold : float
        Seuil utilisé. Par défaut 1/e.

    Returns
    -------
    tau_acf : float
        Temps caractéristique en secondes.
        np.nan si le seuil n'est jamais atteint.
    """

    lag_times = np.asarray(lag_times)
    acf = np.asarray(acf)

    below = np.where(
        acf <= threshold
    )[0]

    # On ne veut pas prendre lag = 0
    below = below[below > 0]

    if len(below) == 0:
        return np.nan

    i = below[0]

    if i == 0:
        return lag_times[0]

    x1 = lag_times[i - 1]
    x2 = lag_times[i]

    y1 = acf[i - 1]
    y2 = acf[i]

    if y2 == y1:
        return x2

    # interpolation linéaire
    tau = x1 + (
        (threshold - y1)
        * (x2 - x1)
        / (y2 - y1)
    )

    return tau

def compute_cumulative_power_frequencies(
    freqs,
    psd,
    percentiles=(0.50, 0.90, 0.95, 0.99)
):
    """
    Calcule les fréquences sous lesquelles se trouve une fraction donnée
    de la puissance spectrale totale.

    Parameters
    ----------
    freqs : np.ndarray
        Fréquences en Hz.
    psd : np.ndarray
        Densité spectrale de puissance moyenne.
    percentiles : tuple
        Fractions cumulées recherchées.
        Par défaut : 50 %, 90 %, 95 %, 99 %.

    Returns
    -------
    results : dict
        Dictionnaire contenant par exemple :
        {
            "f50": ...,
            "f90": ...,
            "f95": ...,
            "f99": ...
        }
    cumulative_fraction : np.ndarray
        Fraction cumulée de puissance pour chaque fréquence.
    """

    freqs = np.asarray(freqs)
    psd = np.asarray(psd)

    if freqs.ndim != 1 or psd.ndim != 1:
        raise ValueError("freqs et psd doivent être des tableaux 1D.")

    if len(freqs) != len(psd):
        raise ValueError("freqs et psd doivent avoir la même longueur.")

    # Intégration cumulative par méthode des trapèzes
    dx = np.diff(freqs)

    areas = 0.5 * (
        psd[:-1] + psd[1:]
    ) * dx

    cumulative_power = np.concatenate([
        [0],
        np.cumsum(areas)
    ])

    total_power = cumulative_power[-1]

    if total_power <= 0:
        raise ValueError("La puissance totale est nulle ou négative.")

    cumulative_fraction = (
        cumulative_power / total_power
    )

    results = {}

    for p in percentiles:

        idx = np.searchsorted(
            cumulative_fraction,
            p
        )

        if idx == 0:
            f_p = freqs[0]

        elif idx >= len(freqs):
            f_p = freqs[-1]

        else:
            # interpolation linéaire
            x1 = cumulative_fraction[idx - 1]
            x2 = cumulative_fraction[idx]

            f1 = freqs[idx - 1]
            f2 = freqs[idx]

            if x2 == x1:
                f_p = f2
            else:
                f_p = f1 + (
                    (p - x1)
                    * (f2 - f1)
                    / (x2 - x1)
                )

        results[f"f{int(p * 100)}"] = f_p

    return results, cumulative_fraction