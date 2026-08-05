import argparse
import gc
import pickle
import time
import traceback

from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from maitrise_curbd.io import load_dataset

from maitrise_curbd.masks import (
    remove_thin_label_artifacts,
    reduce_atlas_to_6_regions,
    subdivide_mask_by_spatial_clustering,
    build_parent_regions_dict,
)

from maitrise_curbd.timeseries import (
    compute_dff,
    extract_timeseries_du_tenseur,
    regress_out_global_signal,
    smooth_timeseries,
)

from maitrise_curbd.curbd import (
    computeCURBD,
    trainMultiRegionRNN,
)


# ============================================================
# PARAMÈTRES DU TEST
# ============================================================

titre_du_test = "test_tauRNN_sigma_C8_M6_409"

now = datetime.now()
maintenant = now.strftime("%Y-%m-%d_%Hh%M")


# Dataset fixe
n_cohorte = 8
month = 6
souris = 409


# Subdivision spatiale fixe
n_pixels = 100


# Paramètres testés
tauRNN_list = [
    0.083,
    0.167,
    0.33,
    0.5,
]

lissage_sigma_list = [
    2,
    4,
    6,
    8,
]


# Prétraitement
use_dff = False
use_global_regression = True


# Paramètres CURBD
dtData = 1 / 12
dtFactor = 4

P0 = 1.0

nRunTrain = 500
nRunFree = 50


# ============================================================
# DOSSIERS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data-dir",
    type=Path,
    default=Path("data"),
)

parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path(
        titre_du_test
    ) / f"run_du_{maintenant}",
)

args = parser.parse_args()

save_dir = args.output_dir
save_dir.mkdir(
    parents=True,
    exist_ok=True,
)

results_csv = save_dir / "night_run_summary.csv"


# ============================================================
# FONCTIONS SAFE POUR LES MÉTRIQUES
# ============================================================

def finite_values(x):
    """
    Retourne seulement les valeurs finies d'un tableau.
    """
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


def safe_nanmax(x):
    """
    Maximum des valeurs finies.
    """
    x = finite_values(x)

    if x.size == 0:
        return np.nan

    return float(np.max(x))


def safe_nanmin(x):
    """
    Minimum des valeurs finies.
    """
    x = finite_values(x)

    if x.size == 0:
        return np.nan

    return float(np.min(x))


def safe_last(x):
    """
    Dernière valeur finie.
    """
    x = finite_values(x)

    if x.size == 0:
        return np.nan

    return float(x[-1])


def get_model_value(model, key, default=None):
    """
    Récupère une valeur dans un modèle qui peut être
    soit un dictionnaire, soit un objet.
    """

    if isinstance(model, dict):
        return model.get(key, default)

    return getattr(model, key, default)


# ============================================================
# CONFIGURATIONS
# ============================================================

configs = list(
    product(
        tauRNN_list,
        lissage_sigma_list,
    )
)

rows = []

print("\n" + "=" * 90)
print("TEST DES PARAMÈTRES CURBD")
print("=" * 90)

print(
    f"Dataset : C{n_cohorte}, M{month}, "
    f"souris {souris}"
)

print(
    f"Nombre de configurations : {len(configs)}"
)

for i, (tauRNN, sigma) in enumerate(configs):

    print(
        f"{i:02d} | "
        f"tauRNN = {tauRNN:.3f} s | "
        f"sigma = {sigma} frames "
        f"({sigma * dtData:.3f} s)"
    )


# ============================================================
# CHARGEMENT DU DATASET
#
# Cette partie est effectuée une seule fois.
# ============================================================

print("\n" + "=" * 90)
print("CHARGEMENT ET PRÉPARATION DU DATASET")
print("=" * 90)

gcamp, atlas, roi_mask = load_dataset(
    cohort=n_cohorte,
    month=month,
    mouse=souris,
)

gcamp = np.asarray(gcamp)
atlas = np.asarray(atlas)
roi_mask = np.asarray(roi_mask)

print(
    f"GCaMP    : shape={gcamp.shape}, dtype={gcamp.dtype}"
)

print(
    f"Atlas    : shape={atlas.shape}, dtype={atlas.dtype}"
)

print(
    f"ROI mask : shape={roi_mask.shape}, dtype={roi_mask.dtype}"
)


# Vérifier que le temps est bien le premier axe
if gcamp.ndim != 3:
    raise ValueError(
        "gcamp doit être un tableau 3D de forme (T, H, W). "
        f"Forme reçue : {gcamp.shape}"
    )

if roi_mask.shape != gcamp.shape[1:]:
    raise ValueError(
        f"roi_mask {roi_mask.shape} incompatible "
        f"avec gcamp {gcamp.shape}"
    )


# ============================================================
# PRÉPARATION DU MASQUE
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

regions = build_parent_regions_dict(
    info_masque_sub
)


labels_valides = np.unique(
    masque_sub[np.isfinite(masque_sub)]
)

print(
    f"Nombre de sous-régions : {len(labels_valides)}"
)


# ============================================================
# EXTRACTION DES SÉRIES TEMPORELLES BRUTES
#
# Le lissage sera appliqué séparément dans chaque configuration.
# ============================================================

ts_raw = extract_timeseries_du_tenseur(
    gcamp,
    masque_sub,
)

ts_raw = np.asarray(
    ts_raw,
    dtype=np.float32,
)

if ts_raw.ndim != 2:
    raise ValueError(
        "ts_raw doit être un tableau 2D. "
        f"Forme reçue : {ts_raw.shape}"
    )

if not np.all(np.isfinite(ts_raw)):
    raise ValueError(
        "ts_raw contient des NaN ou des valeurs infinies."
    )

print(
    f"Séries temporelles brutes : {ts_raw.shape}"
)

print(
    f"Durée approximative : "
    f"{ts_raw.shape[-1] * dtData:.1f} secondes"
)


# On peut libérer les images originales :
# elles ne sont plus nécessaires pendant les entraînements.
del gcamp
del atlas
del roi_mask
del clean_atlas
del atlas_6

gc.collect()


# ============================================================
# BOUCLE PRINCIPALE
# ============================================================

for i_config, (tauRNN, lissage_sigma) in enumerate(configs):

    t0 = time.time()

    save_path = save_dir / (
        f"config{i_config:02d}_"
        f"C{n_cohorte}_"
        f"M{month}_"
        f"mouse{souris}_"
        f"pix{n_pixels}_"
        f"sigma{lissage_sigma}_"
        f"tauRNN{tauRNN:.3f}_"
        f"dff{use_dff}_"
        f"globalreg{use_global_regression}_"
        f"nRunTrain{nRunTrain}.pkl"
    )

    row = {
        "i_config": i_config,

        "cohort": n_cohorte,
        "month": month,
        "mouse": souris,

        "n_pixels": n_pixels,
        "n_regions": np.nan,

        "fps": 1 / dtData,
        "dtData": dtData,
        "dtFactor": dtFactor,

        "lissage_sigma_frames": lissage_sigma,
        "lissage_sigma_sec": (
            lissage_sigma * dtData
        ),
        "lissage_fwhm_sec": (
            2.355 * lissage_sigma * dtData
        ),

        "tauRNN": tauRNN,

        "nRunTrain": nRunTrain,
        "nRunFree": nRunFree,
        "P0": P0,

        "use_dff": use_dff,
        "use_global_regression": (
            use_global_regression
        ),

        "pVar_max": np.nan,
        "pVar_finale": np.nan,

        "chi2_min": np.nan,
        "chi2_final": np.nan,

        "runtime_sec": np.nan,
        "status": "started",
        "error": None,

        "save_path": str(save_path),
    }

    try:

        print("\n" + "=" * 90)

        print(
            f"CONFIGURATION "
            f"{i_config + 1}/{len(configs)}"
        )

        print(
            f"C{n_cohorte} | "
            f"M{month} | "
            f"souris {souris}"
        )

        print(
            f"n_pixels = {n_pixels}"
        )

        print(
            f"sigma = {lissage_sigma} frames "
            f"= {lissage_sigma * dtData:.3f} s"
        )

        print(
            f"tauRNN = {tauRNN:.3f} s"
        )

        print(
            f"nRunTrain = {nRunTrain}"
        )

        print("=" * 90)


        # ----------------------------------------------------
        # Prétraitement propre à cette configuration
        # ----------------------------------------------------

        ts = ts_raw.copy()

        ts = smooth_timeseries(
            ts,
            sigma=lissage_sigma,
        )

        if use_dff:
            ts = compute_dff(ts)

        if use_global_regression:
            ts = regress_out_global_signal(ts)

        ts = np.asarray(
            ts,
            dtype=np.float32,
        )

        if not np.all(np.isfinite(ts)):

            n_nan = np.sum(np.isnan(ts))
            n_inf = np.sum(np.isinf(ts))

            raise ValueError(
                "Les séries temporelles prétraitées "
                "contiennent des valeurs invalides : "
                f"{n_nan} NaN et {n_inf} inf."
            )


        # ----------------------------------------------------
        # Entraînement CURBD
        # ----------------------------------------------------

        model = trainMultiRegionRNN(
            ts,
            dtData=dtData,
            dtFactor=dtFactor,
            tauRNN=tauRNN,
            nRunFree=nRunFree,
            nRunTrain=nRunTrain,
            P0=P0,
            regions=regions,
            plotStatus=False,
        )


        # ----------------------------------------------------
        # Courants CURBD
        # ----------------------------------------------------

        curbd_arr, curbd_labels = computeCURBD(
            model
        )

        n_regions = curbd_arr.shape[0]

        row["n_regions"] = n_regions

        currents_curves = {}

        for iTarget in range(n_regions):

            for iSource in range(n_regions):

                C = curbd_arr[
                    iTarget,
                    iSource,
                ]

                current_curve = np.sum(
                    C,
                    axis=0,
                ).astype(np.float32)

                currents_curves[
                    (iTarget, iSource)
                ] = current_curve


        # ----------------------------------------------------
        # Récupération des résultats du modèle
        # ----------------------------------------------------

        J_final = get_model_value(
            model,
            "J",
        )

        pVar = get_model_value(
            model,
            "pVars",
            np.array([np.nan]),
        )

        chi2 = get_model_value(
            model,
            "chi2s",
            np.array([np.nan]),
        )

        tRNN = get_model_value(
            model,
            "tRNN",
        )

        if J_final is None:
            raise KeyError(
                "Le modèle ne contient pas J."
            )

        if tRNN is None:
            raise KeyError(
                "Le modèle ne contient pas tRNN."
            )

        J_final = np.asarray(
            J_final,
            dtype=np.float32,
        )

        pVar = np.asarray(
            pVar,
            dtype=float,
        )

        chi2 = np.asarray(
            chi2,
            dtype=float,
        )

        tRNN = np.asarray(
            tRNN,
            dtype=np.float32,
        )


        # ----------------------------------------------------
        # Métriques
        # ----------------------------------------------------

        row["pVar_max"] = safe_nanmax(
            pVar
        )

        row["pVar_finale"] = safe_last(
            pVar
        )

        row["chi2_min"] = safe_nanmin(
            chi2
        )

        row["chi2_final"] = safe_last(
            chi2
        )


        # ----------------------------------------------------
        # Sauvegarde
        # ----------------------------------------------------

        row["runtime_sec"] = (
            time.time() - t0
        )

        row["status"] = "done"

        to_save = {
            "J_final": J_final,

            "currents_curves": (
                currents_curves
            ),

            "curbd_labels": curbd_labels,

            "regions": regions,

            "info_masque_sub": (
                info_masque_sub
            ),

            "masque_sub": np.asarray(
                masque_sub,
                dtype=np.float32,
            ),

            "tRNN": tRNN,

            "pVar": pVar,
            "chi2": chi2,

            "parameters": {
                "cohort": n_cohorte,
                "month": month,
                "mouse": souris,

                "n_pixels": n_pixels,

                "fps": 1 / dtData,
                "dtData": dtData,
                "dtFactor": dtFactor,

                "lissage_sigma_frames": (
                    lissage_sigma
                ),

                "lissage_sigma_sec": (
                    lissage_sigma * dtData
                ),

                "lissage_fwhm_sec": (
                    2.355
                    * lissage_sigma
                    * dtData
                ),

                "tauRNN": tauRNN,

                "nRunTrain": nRunTrain,
                "nRunFree": nRunFree,
                "P0": P0,

                "use_dff": use_dff,

                "use_global_regression": (
                    use_global_regression
                ),
            },

            "row": row.copy(),
        }

        with open(save_path, "wb") as f:

            pickle.dump(
                to_save,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


        print(
            f"✅ Sauvegarde réussie : "
            f"{save_path}"
        )

        print(
            f"n_regions = {n_regions}"
        )

        print(
            "Nombre de courbes sauvegardées "
            f"= {len(currents_curves)}"
        )

        print(
            f"pVar max = "
            f"{row['pVar_max']:.4f}"
        )

        print(
            f"pVar finale = "
            f"{row['pVar_finale']:.4f}"
        )

        print(
            f"chi2 min = "
            f"{row['chi2_min']:.4f}"
        )

        print(
            f"chi2 final = "
            f"{row['chi2_final']:.4f}"
        )

        print(
            f"runtime = "
            f"{row['runtime_sec']:.1f} s"
        )


    except Exception:

        row["status"] = "failed"

        row["error"] = (
            traceback.format_exc()
        )

        row["runtime_sec"] = (
            time.time() - t0
        )

        print("❌ ERREUR")

        print(
            row["error"]
        )


    finally:

        rows.append(
            row.copy()
        )

        pd.DataFrame(
            rows
        ).to_csv(
            results_csv,
            index=False,
        )

        print(
            f"CSV résumé mis à jour : "
            f"{results_csv}"
        )


        # On supprime seulement les objets propres
        # à cette configuration.
        for variable_name in [
            "ts",
            "model",
            "curbd_arr",
            "curbd_labels",
            "currents_curves",
            "current_curve",
            "C",
            "J_final",
            "pVar",
            "chi2",
            "tRNN",
            "to_save",
        ]:

            globals().pop(
                variable_name,
                None,
            )

        gc.collect()


# ============================================================
# FIN
# ============================================================

print("\n" + "=" * 90)
print("NIGHT RUN TERMINÉE")
print(f"Résumé CSV : {results_csv}")
print("=" * 90)