
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from maitrise_curbd.io import load_dataset
from maitrise_curbd.masks import (
    remove_thin_label_artifacts,
    reduce_atlas_to_6_regions,
    subdivide_mask_by_spatial_clustering,
)
from maitrise_curbd.timeseries import (
    compute_dff,
    extract_timeseries_du_tenseur,
    regress_out_global_signal,
    smooth_timeseries,
)


# ============================================================
# PARAMÈTRES
# ============================================================

titre_du_test = "benchmark_AR_predictibilite"

# Dataset
n_cohorte = 8
month = 6
souris = 409

# Même résolution que notre meilleur pipeline actuel
n_pixels = 100

# Prétraitement
dtData = 1 / 12
fs = 1 / dtData

use_dff = True
use_global_regression = True
lissage_sigma = 2

# Benchmark autoregressif
#
# Chaque trace est prédite à partir de SES PROPRES valeurs passées.
#
# À 12 Hz :
# lag=1   -> 0.083 s de mémoire
# lag=2   -> 0.167 s
# lag=4   -> 0.333 s
# lag=6   -> 0.500 s
# lag=12  -> 1.000 s
# lag=24  -> 2.000 s
lags_list = [
    1,
    2,
    4,
    6,
    12,
    24,
]

# Ridge : lambda
ridge_list = [
    0.0,
    1e-6,
    1e-4,
    1e-2,
    1e-1,
    1.0,
]

train_fraction = 0.70


# ============================================================
# DOSSIERS
# ============================================================

now = datetime.now()
maintenant = now.strftime("%Y-%m-%d_%Hh%M")

parser = argparse.ArgumentParser()

parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path(titre_du_test) / f"run_du_{maintenant}",
)

args = parser.parse_args()

save_dir = args.output_dir
save_dir.mkdir(
    parents=True,
    exist_ok=True,
)

results_csv = save_dir / "AR_benchmark_summary.csv"


# ============================================================
# OUTILS
# ============================================================

def make_ar_design_matrix(trace, n_lags):
    """
    Construit :
        X[t] = [x(t-1), x(t-2), ..., x(t-n_lags)]
        y[t] = x(t)

    Retour
    ------
    X : (T-n_lags, n_lags)
    y : (T-n_lags,)
    """

    trace = np.asarray(trace, dtype=float)

    if trace.ndim != 1:
        raise ValueError("trace doit être 1D.")

    if len(trace) <= n_lags:
        raise ValueError(
            f"Trace trop courte ({len(trace)}) pour {n_lags} lags."
        )

    y = trace[n_lags:]

    X = np.column_stack(
        [
            trace[n_lags - lag: -lag]
            for lag in range(1, n_lags + 1)
        ]
    )

    return X, y


def fit_ridge_closed_form(X, y, ridge_lambda):
    """
    Régression ridge avec intercept non pénalisé.

    y = b0 + X beta
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # Standardisation des prédicteurs basée uniquement sur le train
    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)

    x_std = np.where(
        x_std < 1e-12,
        1.0,
        x_std,
    )

    Xz = (X - x_mean) / x_std

    y_mean = np.mean(y)
    yz = y - y_mean

    p = Xz.shape[1]

    A = Xz.T @ Xz

    if ridge_lambda > 0:
        A = A + ridge_lambda * np.eye(p)

    b = Xz.T @ yz

    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(A) @ b

    return {
        "beta": beta,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
    }


def predict_ridge(model, X):
    X = np.asarray(X, dtype=float)

    Xz = (
        (X - model["x_mean"])
        / model["x_std"]
    )

    return (
        model["y_mean"]
        + Xz @ model["beta"]
    )


def pvar_score(y_true, y_pred):
    """
    Même idée générale que pVar :
        1 - SSE/SST

    Ici SST est calculé autour de la moyenne des vraies valeurs
    de l'ensemble évalué.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if ss_tot <= 0:
        return np.nan

    return 1 - ss_res / ss_tot


# ============================================================
# CHARGEMENT
# ============================================================

print("\n" + "=" * 90)
print("BENCHMARK AUTOREGRESSIF")
print("=" * 90)

print(
    f"Dataset : C{n_cohorte}, M{month}, souris {souris}"
)

gcamp, atlas, roi_mask = load_dataset(
    cohort=n_cohorte,
    month=month,
    mouse=souris,
)

gcamp = np.asarray(gcamp)
atlas = np.asarray(atlas)
roi_mask = np.asarray(roi_mask)

print(
    f"GCaMP : {gcamp.shape}"
)


# ============================================================
# MASQUE / SOUS-RÉGIONS
# ============================================================

clean_atlas = remove_thin_label_artifacts(
    atlas,
    size=5,
    min_fraction=0.25,
)

atlas_6 = reduce_atlas_to_6_regions(
    atlas=clean_atlas,
    roi_mask=roi_mask,
)

masque_sub, info_masque_sub = (
    subdivide_mask_by_spatial_clustering(
        atlas_6,
        target_size=n_pixels,
    )
)

labels_valides = np.unique(
    masque_sub[np.isfinite(masque_sub)]
)

print(
    f"Nombre de sous-régions : {len(labels_valides)}"
)


# ============================================================
# EXTRACTION + MÊME PREPROCESSING QUE CURBD
# ============================================================

ts = extract_timeseries_du_tenseur(
    gcamp,
    masque_sub,
)

ts = np.asarray(
    ts,
    dtype=float,
)

if use_dff:
    ts = compute_dff(
        ts,
        fs=fs,
        window_sec=60,
        percentile=8,
    )

if use_global_regression:
    ts = regress_out_global_signal(
        ts
    )

ts = smooth_timeseries(
    ts,
    sigma=lissage_sigma,
)

if not np.all(np.isfinite(ts)):
    raise ValueError(
        "Les séries temporelles prétraitées contiennent "
        "des NaN ou des inf."
    )

print(
    f"Shape finale : {ts.shape}"
)

print(
    f"Durée : {ts.shape[1] * dtData:.1f} s"
)


# ============================================================
# BENCHMARK
# ============================================================

rows = []

for n_lags in lags_list:

    memory_sec = n_lags * dtData

    print("\n" + "-" * 90)
    print(
        f"{n_lags} lags = {memory_sec:.3f} s de passé"
    )

    # Toutes les traces auront le même nombre d'échantillons
    n_samples = ts.shape[1] - n_lags

    split_index = int(
        train_fraction * n_samples
    )

    if split_index <= 0 or split_index >= n_samples:
        raise ValueError(
            "train_fraction produit une séparation invalide."
        )

    for ridge_lambda in ridge_list:

        all_train_true = []
        all_train_pred = []

        all_test_true = []
        all_test_pred = []

        pvar_train_by_trace = []
        pvar_test_by_trace = []

        for i_trace in range(ts.shape[0]):

            X, y = make_ar_design_matrix(
                ts[i_trace],
                n_lags=n_lags,
            )

            X_train = X[:split_index]
            y_train = y[:split_index]

            X_test = X[split_index:]
            y_test = y[split_index:]

            model = fit_ridge_closed_form(
                X_train,
                y_train,
                ridge_lambda=ridge_lambda,
            )

            y_pred_train = predict_ridge(
                model,
                X_train,
            )

            y_pred_test = predict_ridge(
                model,
                X_test,
            )

            pvar_train_trace = pvar_score(
                y_train,
                y_pred_train,
            )

            pvar_test_trace = pvar_score(
                y_test,
                y_pred_test,
            )

            pvar_train_by_trace.append(
                pvar_train_trace
            )

            pvar_test_by_trace.append(
                pvar_test_trace
            )

            all_train_true.append(
                y_train
            )

            all_train_pred.append(
                y_pred_train
            )

            all_test_true.append(
                y_test
            )

            all_test_pred.append(
                y_pred_test
            )


        # Agrégation globale sur toutes les traces
        all_train_true = np.concatenate(
            all_train_true
        )

        all_train_pred = np.concatenate(
            all_train_pred
        )

        all_test_true = np.concatenate(
            all_test_true
        )

        all_test_pred = np.concatenate(
            all_test_pred
        )

        pvar_train_global = pvar_score(
            all_train_true,
            all_train_pred,
        )

        pvar_test_global = pvar_score(
            all_test_true,
            all_test_pred,
        )

        pvar_train_by_trace = np.asarray(
            pvar_train_by_trace,
            dtype=float,
        )

        pvar_test_by_trace = np.asarray(
            pvar_test_by_trace,
            dtype=float,
        )

        row = {
            "cohort": n_cohorte,
            "month": month,
            "mouse": souris,

            "n_pixels": n_pixels,
            "n_subregions": ts.shape[0],

            "fps": fs,
            "sigma_frames": lissage_sigma,
            "sigma_sec": lissage_sigma * dtData,

            "n_lags": n_lags,
            "memory_sec": memory_sec,
            "ridge_lambda": ridge_lambda,

            "train_fraction": train_fraction,
            "n_train_samples_per_trace": split_index,
            "n_test_samples_per_trace": (
                n_samples - split_index
            ),

            "pVar_train_global": (
                pvar_train_global
            ),

            "pVar_test_global": (
                pvar_test_global
            ),

            "pVar_train_trace_mean": (
                np.nanmean(
                    pvar_train_by_trace
                )
            ),

            "pVar_train_trace_median": (
                np.nanmedian(
                    pvar_train_by_trace
                )
            ),

            "pVar_test_trace_mean": (
                np.nanmean(
                    pvar_test_by_trace
                )
            ),

            "pVar_test_trace_median": (
                np.nanmedian(
                    pvar_test_by_trace
                )
            ),

            "pVar_test_trace_p10": (
                np.nanpercentile(
                    pvar_test_by_trace,
                    10,
                )
            ),

            "pVar_test_trace_p90": (
                np.nanpercentile(
                    pvar_test_by_trace,
                    90,
                )
            ),
        }

        rows.append(row)

        print(
            f"ridge={ridge_lambda:<8g} | "
            f"pVar train={pvar_train_global:.4f} | "
            f"pVar TEST={pvar_test_global:.4f}"
        )


# ============================================================
# SAUVEGARDE / RÉSUMÉ
# ============================================================

results_df = pd.DataFrame(
    rows
)

results_df.to_csv(
    results_csv,
    index=False,
)

valid = results_df[
    np.isfinite(
        results_df["pVar_test_global"]
    )
].copy()

valid = valid.sort_values(
    "pVar_test_global",
    ascending=False,
)

print("\n" + "=" * 90)
print("TOP 10 — PRÉDICTIBILITÉ HORS ÉCHANTILLON")
print("=" * 90)

print(
    valid[
        [
            "n_lags",
            "memory_sec",
            "ridge_lambda",
            "pVar_train_global",
            "pVar_test_global",
            "pVar_test_trace_median",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

if not valid.empty:

    best = valid.iloc[0]

    print("\nMEILLEUR BENCHMARK")
    print(
        f"Mémoire = {best['memory_sec']:.3f} s "
        f"({int(best['n_lags'])} lags)"
    )

    print(
        f"ridge = {best['ridge_lambda']}"
    )

    print(
        f"pVar train = "
        f"{best['pVar_train_global']:.4f}"
    )

    print(
        f"pVar TEST = "
        f"{best['pVar_test_global']:.4f}"
    )

print(
    f"\nCSV : {results_csv}"
)

print("=" * 90)
