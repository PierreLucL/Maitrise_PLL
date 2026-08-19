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
#Prochain test tauRNN= 0.3, dtFactor = 2, sigma = 4 et 


titre_du_test = "FAST_grid_g_ampInWN_tauRNN033_dtFactor2_sigma2"

now = datetime.now()
maintenant = now.strftime("%Y-%m-%d_%Hh%M")


# Dataset fixe
n_cohorte = 8
month = 6
souris = 409


# Subdivision spatiale fixe
n_pixels = 100


# Paramètres testés
g_list = [
    0.8,
    1.0,
    1.5,
    2.0,
]

ampInWN_list = [
    0.0,
    0.001,
    0.003,
    0.01,
]

# Paramètres fixés
tauRNN = 0.33
dtFactor = 2
P0 = 1.0

# Lissage
lissage_sigma = 2

# Prétraitement
use_dff = True
use_global_regression = True

# Paramètres CURBD
dtData = 1 / 12
tauWN = 0.1

# Screening rapide
nRunTrain = 200
nRunFree = 5


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


def safe_nanmean(x):
    """
    Moyenne des valeurs finies.
    """
    x = finite_values(x)

    if x.size == 0:
        return np.nan

    return float(np.mean(x))


def safe_nanstd(x):
    """
    Écart-type des valeurs finies.
    """
    x = finite_values(x)

    if x.size == 0:
        return np.nan

    return float(np.std(x))


def safe_index(x, idx):
    """
    Valeur à un indice donné si elle existe et est finie.
    """
    x = np.asarray(x, dtype=float).ravel()

    if x.size == 0:
        return np.nan

    if idx < 0:
        idx = x.size + idx

    if idx < 0 or idx >= x.size:
        return np.nan

    value = x[idx]

    if not np.isfinite(value):
        return np.nan

    return float(value)


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
        g_list,
        ampInWN_list,
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

for i, (g, ampInWN) in enumerate(configs):

    dtRNN = dtData / dtFactor
    alpha_dt_tau = dtRNN / tauRNN

    print(
        f"{i:02d} | "
        f"g = {g:.3f} | "
        f"ampInWN = {ampInWN:.4f} | "
        f"P0 = {P0:.3f} | "
        f"tauRNN = {tauRNN:.3f} s | "
        f"dtFactor = {dtFactor} | "
        f"alpha = {alpha_dt_tau:.4f}"
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


# ============================================================
# PRÉTRAITEMENT COMMUN À TOUTES LES CONFIGURATIONS
#
# Important :
#   fluorescence régionale brute
#   -> ΔF/F avec la vraie fréquence d'acquisition
#   -> régression du signal global
#   -> lissage (appliqué ensuite, juste avant CURBD)
# ============================================================

ts_preprocessed = ts_raw.copy()

if use_dff:
    ts_preprocessed = compute_dff(
        ts_preprocessed,
        fs=1 / dtData,
        window_sec=60,
        percentile=8,
    )

if use_global_regression:
    ts_preprocessed = regress_out_global_signal(
        ts_preprocessed
    )

ts_preprocessed = np.asarray(
    ts_preprocessed,
    dtype=np.float32,
)

if not np.all(np.isfinite(ts_preprocessed)):
    n_nan = np.sum(np.isnan(ts_preprocessed))
    n_inf = np.sum(np.isinf(ts_preprocessed))

    raise ValueError(
        "Les séries temporelles après ΔF/F / GSR "
        f"contiennent {n_nan} NaN et {n_inf} inf."
    )


# ------------------------------------------------------------
# Diagnostic de normalisation / clipping dans CURBD
# ------------------------------------------------------------

curbd_scale = np.max(ts_preprocessed)

if not np.isfinite(curbd_scale) or curbd_scale <= 0:
    raise ValueError(
        "Le maximum des séries temporelles doit être positif "
        "pour la normalisation interne de CURBD."
    )

scaled_for_curbd = ts_preprocessed / curbd_scale

fraction_clip_pos = np.mean(
    scaled_for_curbd > 0.999
)

fraction_clip_neg = np.mean(
    scaled_for_curbd < -0.999
)

print("\nDiagnostic de normalisation CURBD")
print(f"Maximum : {np.max(ts_preprocessed)}")
print(f"Minimum : {np.min(ts_preprocessed)}")
print(f"% > 0.999 : {100 * fraction_clip_pos}")
print(f"% < -0.999 : {100 * fraction_clip_neg}")


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

for i_config, (g, ampInWN) in enumerate(configs):

    t0 = time.time()

    save_path = save_dir / (
        f"config{i_config:02d}_"
        f"C{n_cohorte}_"
        f"M{month}_"
        f"mouse{souris}_"
        f"pix{n_pixels}_"
        f"sigma{lissage_sigma}_"
        f"tauRNN{tauRNN:.3f}_"
        f"dtFactor{dtFactor}_"
        f"g{g:.2f}_"
        f"ampWN{ampInWN:.4f}_"
        f"P0{P0:.2f}_"
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
        "dtRNN": dtData / dtFactor,
        "alpha_dt_tau": (dtData / dtFactor) / tauRNN,

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

        "g": g,
        "tauWN": tauWN,
        "ampInWN": ampInWN,
        "P0": P0,

        "use_dff": use_dff,
        "use_global_regression": (
            use_global_regression
        ),

        # Métriques globales (compatibilité avec les anciens CSV)
        "pVar_max": np.nan,
        "pVar_finale": np.nan,
        "chi2_min": np.nan,
        "chi2_final": np.nan,

        # Métriques séparant entraînement et runs libres
        "pVar_max_train": np.nan,
        "pVar_train_end": np.nan,
        "pVar_free_mean": np.nan,
        "pVar_free_std": np.nan,
        "pVar_free_min": np.nan,
        "pVar_free_final": np.nan,

        "chi2_min_train": np.nan,
        "chi2_train_end": np.nan,
        "chi2_free_mean": np.nan,
        "chi2_free_std": np.nan,
        "chi2_free_min": np.nan,
        "chi2_free_final": np.nan,

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
            f"dtFactor = {dtFactor} | "
            f"dtRNN = {dtData / dtFactor:.5f} s | "
            f"alpha = {(dtData / dtFactor) / tauRNN:.4f}"
        )

        print(
            f"g = {g:.3f} | "
            f"ampInWN = {ampInWN:.4f} | "
            f"P0 = {P0:.3f}"
        )

        print(
            f"nRunTrain = {nRunTrain}"
        )

        print("=" * 90)


        # ----------------------------------------------------
        # Prétraitement propre à cette configuration
        # ----------------------------------------------------

        # Le ΔF/F et la régression globale ont déjà été faits
        # une seule fois, avant la boucle. Ici, seule la dernière
        # étape (lissage) est appliquée au signal donné à CURBD.
        ts = smooth_timeseries(
            ts_preprocessed,
            sigma=lissage_sigma,
        )

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
            g=g,
            tauRNN=tauRNN,
            tauWN=tauWN,
            ampInWN=ampInWN,
            nRunFree=nRunFree,
            nRunTrain=nRunTrain,
            P0=P0,
            regions=regions,
            plotStatus=False,
        )
        # ----------------------------------------------------
        # Courants CURBD
        # ----------------------------------------------------
        # SCREENING RAPIDE :
        # on ne calcule pas computeCURBD() ici.
        # Le but est seulement de comparer les hyperparamètres
        # avec pVar / chi2 et la stabilité des runs libres.
        curbd_arr = None
        curbd_labels = None
        currents_curves = None

        # Nombre de sous-régions du modèle
        n_regions = len(regions)
        row["n_regions"] = n_regions


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

        # Métriques globales (pour compatibilité avec les anciens CSV)
        row["pVar_max"] = safe_nanmax(pVar)
        row["pVar_finale"] = safe_last(pVar)

        row["chi2_min"] = safe_nanmin(chi2)
        row["chi2_final"] = safe_last(chi2)


        # Séparation entraînement / runs libres
        pVar_train = pVar[:nRunTrain]
        pVar_free = pVar[nRunTrain:nRunTrain + nRunFree]

        chi2_train = chi2[:nRunTrain]
        chi2_free = chi2[nRunTrain:nRunTrain + nRunFree]


        row["pVar_max_train"] = safe_nanmax(
            pVar_train
        )

        row["pVar_train_end"] = safe_index(
            pVar,
            nRunTrain - 1,
        )

        row["pVar_free_mean"] = safe_nanmean(
            pVar_free
        )

        row["pVar_free_std"] = safe_nanstd(
            pVar_free
        )

        row["pVar_free_min"] = safe_nanmin(
            pVar_free
        )

        row["pVar_free_final"] = safe_last(
            pVar_free
        )


        row["chi2_min_train"] = safe_nanmin(
            chi2_train
        )

        row["chi2_train_end"] = safe_index(
            chi2,
            nRunTrain - 1,
        )

        row["chi2_free_mean"] = safe_nanmean(
            chi2_free
        )

        row["chi2_free_std"] = safe_nanstd(
            chi2_free
        )

        row["chi2_free_min"] = safe_nanmin(
            chi2_free
        )

        row["chi2_free_final"] = safe_last(
            chi2_free
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
            "regions": regions,
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
                "dtRNN": dtData / dtFactor,
                "alpha_dt_tau": (
                    (dtData / dtFactor) / tauRNN
                ),
                "lissage_sigma_frames": lissage_sigma,
                "lissage_sigma_sec": lissage_sigma * dtData,
                "lissage_fwhm_sec": (
                    2.355 * lissage_sigma * dtData
                ),
                "tauRNN": tauRNN,
                "nRunTrain": nRunTrain,
                "nRunFree": nRunFree,
                "g": g,
                "tauWN": tauWN,
                "ampInWN": ampInWN,
                "P0": P0,
                "use_dff": use_dff,
                "use_global_regression": use_global_regression,
                "fraction_clip_pos": float(fraction_clip_pos),
                "fraction_clip_neg": float(fraction_clip_neg),
                "screening_fast": True,
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
        print("Mode screening rapide : courants CURBD non calculés")

        print(
            f"pVar max = "
            f"{row['pVar_max']:.4f}"
        )

        print(
            f"pVar finale = "
            f"{row['pVar_finale']:.4f}"
        )


        print(
            f"pVar fin entraînement = "
            f"{row['pVar_train_end']:.4f}"
        )

        print(
            f"pVar runs libres = "
            f"{row['pVar_free_mean']:.4f} "
            f"± {row['pVar_free_std']:.4f}"
        )

        print(
            f"pVar min runs libres = "
            f"{row['pVar_free_min']:.4f}"
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
            "J_final",
            "pVar",
            "chi2",
            "pVar_train",
            "pVar_free",
            "chi2_train",
            "chi2_free",
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

results_df = pd.DataFrame(rows)
done_df = results_df[results_df["status"] == "done"].copy()

if not done_df.empty:
    valid = done_df[np.isfinite(done_df["pVar_free_mean"])].copy()

    if not valid.empty:
        valid = valid.sort_values(
            "pVar_free_mean",
            ascending=False,
        )

        print("\nTOP CONFIGURATIONS")
        for rank, (_, best) in enumerate(
            valid.head(5).iterrows(),
            start=1,
        ):
            print(
                f"{rank}. "
                f"g={best['g']:.3f} | "
                f"ampInWN={best['ampInWN']:.4f} | "
                f"pVar libre={best['pVar_free_mean']:.4f} | "
                f"pVar train end={best['pVar_train_end']:.4f} | "
                f"chi2 libre={best['chi2_free_mean']:.4f}"
            )

print("=" * 90)
