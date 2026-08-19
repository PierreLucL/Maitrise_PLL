
import argparse
import gc
import pickle
import time
import traceback

from datetime import datetime
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
    trainMultiRegionRNN,
)


# ============================================================
# DATASETS À ENTRAÎNER
# ============================================================
#
# Format :
#     (cohorte, mois, souris)
#
# Ajoute simplement les autres souris ici.
#
datasets = [
    # C2
    (2, 6, 308),

    # C3
    (3, 6, 316),
    (3, 6, 322),

    # C5
    (5, 6, 353),
    (5, 6, 361),

    # C6
    (6, 6, 365),
    (6, 6, 367),
    (6, 6, 374),

    # C7
    (7, 6, 387),
    (7, 6, 396),
    (7, 6, 397),

    # C8
    (8, 6, 409),

    # C9
    (9, 6, 410),
    (9, 6, 415),
]


# ============================================================
# PARAMÈTRES FIXÉS
# ============================================================

titre_du_test = "TRAIN_M6_toutes_souris_parametres_fixes"

# Résolution spatiale retenue pour l'instant
n_pixels = 100

# Prétraitement
use_dff = True
use_global_regression = True
lissage_sigma = 2

# Sampling
dtData = 1 / 12

# CURBD
tauRNN = 0.33
dtFactor = 2

g = 0.8
P0 = 1.0

tauWN = 0.1
ampInWN = 0.01

# Screening inter-souris
#
# Mets 1000 / 50 plus tard pour la validation finale.
nRunTrain = 200
nRunFree = 5


# ============================================================
# DOSSIERS
# ============================================================

now = datetime.now()
maintenant = now.strftime("%Y-%m-%d_%Hh%M")

parser = argparse.ArgumentParser()

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

results_csv = (
    save_dir
    / "multisouris_train_summary.csv"
)


# ============================================================
# FONCTIONS SAFE
# ============================================================

def finite_values(x):
    x = np.asarray(
        x,
        dtype=float,
    ).ravel()

    return x[
        np.isfinite(x)
    ]


def safe_nanmax(x):

    x = finite_values(x)

    if x.size == 0:
        return np.nan

    return float(
        np.max(x)
    )


def safe_nanmin(x):

    x = finite_values(x)

    if x.size == 0:
        return np.nan

    return float(
        np.min(x)
    )


def safe_nanmean(x):

    x = finite_values(x)

    if x.size == 0:
        return np.nan

    return float(
        np.mean(x)
    )


def safe_nanstd(x):

    x = finite_values(x)

    if x.size == 0:
        return np.nan

    return float(
        np.std(x)
    )


def safe_last(x):

    x = finite_values(x)

    if x.size == 0:
        return np.nan

    return float(
        x[-1]
    )


def safe_index(x, idx):

    x = np.asarray(
        x,
        dtype=float,
    ).ravel()

    if x.size == 0:
        return np.nan

    if idx < 0:
        idx = (
            x.size
            + idx
        )

    if (
        idx < 0
        or idx >= x.size
    ):
        return np.nan

    value = x[idx]

    if not np.isfinite(
        value
    ):
        return np.nan

    return float(
        value
    )


# ============================================================
# INFO GÉNÉRALE
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "TRAIN CURBD MULTI-SOURIS"
)

print(
    "=" * 100
)

print(
    f"Nombre de souris : "
    f"{len(datasets)}"
)

print(
    f"n_pixels = {n_pixels}"
)

print(
    f"sigma = {lissage_sigma}"
)

print(
    f"tauRNN = {tauRNN}"
)

print(
    f"dtFactor = {dtFactor}"
)

print(
    f"g = {g}"
)

print(
    f"ampInWN = {ampInWN}"
)

print(
    f"P0 = {P0}"
)

print(
    f"nRunTrain = {nRunTrain}"
)

print(
    f"nRunFree = {nRunFree}"
)


# ============================================================
# BOUCLE SUR LES SOURIS
# ============================================================

rows = []

for i_dataset, (
    n_cohorte,
    month,
    souris,
) in enumerate(datasets):

    t0 = time.time()

    save_path = (
        save_dir
        / (
            f"mouse{i_dataset:02d}_"
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
            f"nRunTrain{nRunTrain}.pkl"
        )
    )


    row = {

        "i_dataset": i_dataset,

        "cohort": n_cohorte,
        "month": month,
        "mouse": souris,

        "n_pixels": n_pixels,

        "n_subregions": np.nan,
        "n_parent_regions": np.nan,

        "fps": 1 / dtData,

        "dtData": dtData,
        "dtFactor": dtFactor,

        "dtRNN": (
            dtData
            / dtFactor
        ),

        "alpha_dt_tau": (
            (
                dtData
                / dtFactor
            )
            / tauRNN
        ),

        "lissage_sigma_frames": (
            lissage_sigma
        ),

        "lissage_sigma_sec": (
            lissage_sigma
            * dtData
        ),

        "tauRNN": tauRNN,

        "g": g,
        "P0": P0,

        "tauWN": tauWN,
        "ampInWN": ampInWN,

        "nRunTrain": nRunTrain,
        "nRunFree": nRunFree,

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

        "fraction_clip_pos": np.nan,
        "fraction_clip_neg": np.nan,

        "runtime_sec": np.nan,

        "status": "started",

        "error": None,

        "save_path": str(
            save_path
        ),
    }


    try:

        print(
            "\n"
            + "=" * 100
        )

        print(
            f"SOURIS "
            f"{i_dataset + 1}"
            f"/{len(datasets)}"
        )

        print(
            f"C{n_cohorte} | "
            f"M{month} | "
            f"souris {souris}"
        )

        print(
            "=" * 100
        )


        # ====================================================
        # CHARGEMENT
        # ====================================================

        gcamp, atlas, roi_mask = load_dataset(
            cohort=n_cohorte,
            month=month,
            mouse=souris,
        )

        gcamp = np.asarray(
            gcamp
        )

        atlas = np.asarray(
            atlas
        )

        roi_mask = np.asarray(
            roi_mask
        )

        print(
            f"GCaMP : "
            f"{gcamp.shape}"
        )


        # ====================================================
        # MASQUE
        # ====================================================

        clean_atlas = remove_thin_label_artifacts(
            atlas,
            size=5,
            min_fraction=0.25,
        )

        atlas_6 = reduce_atlas_to_6_regions(
            atlas=clean_atlas,
            roi_mask=roi_mask,
        )

        (
            masque_sub,
            info_masque_sub,
        ) = (
            subdivide_mask_by_spatial_clustering(
                atlas_6,
                target_size=n_pixels,
            )
        )

        regions = (
            build_parent_regions_dict(
                info_masque_sub
            )
        )


        labels_valides = np.unique(
            masque_sub[
                np.isfinite(
                    masque_sub
                )
            ]
        )

        n_subregions = len(
            labels_valides
        )

        n_parent_regions = len(
            regions
        )

        row[
            "n_subregions"
        ] = n_subregions

        row[
            "n_parent_regions"
        ] = n_parent_regions


        print(
            f"Sous-régions : "
            f"{n_subregions}"
        )

        print(
            f"Régions parentes : "
            f"{n_parent_regions}"
        )


        # ====================================================
        # EXTRACTION DES TS
        # ====================================================

        ts = (
            extract_timeseries_du_tenseur(
                gcamp,
                masque_sub,
            )
        )

        ts = np.asarray(
            ts,
            dtype=np.float32,
        )


        if not np.all(
            np.isfinite(ts)
        ):

            raise ValueError(
                "ts brut contient "
                "des NaN ou inf."
            )


        # ====================================================
        # PREPROCESSING
        # ====================================================

        if use_dff:

            ts = compute_dff(
                ts,
                fs=1 / dtData,
                window_sec=60,
                percentile=8,
            )


        if use_global_regression:

            ts = (
                regress_out_global_signal(
                    ts
                )
            )


        ts = smooth_timeseries(
            ts,
            sigma=lissage_sigma,
        )


        ts = np.asarray(
            ts,
            dtype=np.float32,
        )


        if not np.all(
            np.isfinite(ts)
        ):

            raise ValueError(
                "TS prétraitées "
                "contiennent NaN/inf."
            )


        # ====================================================
        # DIAGNOSTIC CLIPPING CURBD
        # ====================================================

        curbd_scale = np.max(
            ts
        )

        if (
            not np.isfinite(
                curbd_scale
            )
            or curbd_scale <= 0
        ):

            raise ValueError(
                "Maximum invalide "
                "avant CURBD."
            )


        scaled_for_curbd = (
            ts
            / curbd_scale
        )

        fraction_clip_pos = np.mean(
            scaled_for_curbd
            > 0.999
        )

        fraction_clip_neg = np.mean(
            scaled_for_curbd
            < -0.999
        )


        row[
            "fraction_clip_pos"
        ] = float(
            fraction_clip_pos
        )

        row[
            "fraction_clip_neg"
        ] = float(
            fraction_clip_neg
        )


        print(
            f"Clipping + : "
            f"{100 * fraction_clip_pos:.6f}%"
        )

        print(
            f"Clipping - : "
            f"{100 * fraction_clip_neg:.6f}%"
        )


        # ====================================================
        # TRAIN CURBD
        # ====================================================

        model = trainMultiRegionRNN(
            ts,

            dtData=dtData,

            dtFactor=dtFactor,

            g=g,

            tauRNN=tauRNN,

            tauWN=tauWN,

            ampInWN=ampInWN,

            nRunTrain=nRunTrain,

            nRunFree=nRunFree,

            P0=P0,

            regions=regions,

            plotStatus=False,
        )


        # ====================================================
        # RÉSULTATS
        # ====================================================

        pVar = np.asarray(
            model["pVars"],
            dtype=float,
        )

        chi2 = np.asarray(
            model["chi2s"],
            dtype=float,
        )

        J_final = np.asarray(
            model["J"],
            dtype=np.float32,
        )


        pVar_train = (
            pVar[:nRunTrain]
        )

        pVar_free = (
            pVar[
                nRunTrain:
                nRunTrain + nRunFree
            ]
        )


        chi2_train = (
            chi2[:nRunTrain]
        )

        chi2_free = (
            chi2[
                nRunTrain:
                nRunTrain + nRunFree
            ]
        )


        row[
            "pVar_max_train"
        ] = safe_nanmax(
            pVar_train
        )

        row[
            "pVar_train_end"
        ] = safe_index(
            pVar,
            nRunTrain - 1,
        )

        row[
            "pVar_free_mean"
        ] = safe_nanmean(
            pVar_free
        )

        row[
            "pVar_free_std"
        ] = safe_nanstd(
            pVar_free
        )

        row[
            "pVar_free_min"
        ] = safe_nanmin(
            pVar_free
        )

        row[
            "pVar_free_final"
        ] = safe_last(
            pVar_free
        )


        row[
            "chi2_min_train"
        ] = safe_nanmin(
            chi2_train
        )

        row[
            "chi2_train_end"
        ] = safe_index(
            chi2,
            nRunTrain - 1,
        )

        row[
            "chi2_free_mean"
        ] = safe_nanmean(
            chi2_free
        )

        row[
            "chi2_free_std"
        ] = safe_nanstd(
            chi2_free
        )

        row[
            "chi2_free_min"
        ] = safe_nanmin(
            chi2_free
        )

        row[
            "chi2_free_final"
        ] = safe_last(
            chi2_free
        )


        row[
            "runtime_sec"
        ] = (
            time.time()
            - t0
        )

        row[
            "status"
        ] = "done"


        # ====================================================
        # SAUVEGARDE
        # ====================================================

        to_save = {

            "J_final": J_final,

            "pVar": pVar,

            "chi2": chi2,

            "regions": regions,

            "parameters": {

                "cohort": n_cohorte,
                "month": month,
                "mouse": souris,

                "n_pixels": n_pixels,

                "n_subregions": (
                    n_subregions
                ),

                "n_parent_regions": (
                    n_parent_regions
                ),

                "dtData": dtData,
                "dtFactor": dtFactor,

                "tauRNN": tauRNN,

                "g": g,
                "P0": P0,

                "tauWN": tauWN,
                "ampInWN": ampInWN,

                "lissage_sigma": (
                    lissage_sigma
                ),

                "nRunTrain": (
                    nRunTrain
                ),

                "nRunFree": (
                    nRunFree
                ),
            },

            "row": row.copy(),
        }


        with open(
            save_path,
            "wb",
        ) as f:

            pickle.dump(
                to_save,
                f,
                protocol=(
                    pickle.HIGHEST_PROTOCOL
                ),
            )


        print(
            f"✅ pVar fin train : "
            f"{row['pVar_train_end']:.4f}"
        )

        print(
            f"✅ pVar libre : "
            f"{row['pVar_free_mean']:.4f} "
            f"± "
            f"{row['pVar_free_std']:.4f}"
        )

        print(
            f"✅ chi2 libre : "
            f"{row['chi2_free_mean']:.4f}"
        )

        print(
            f"✅ runtime : "
            f"{row['runtime_sec']/60:.1f} min"
        )


    except Exception:

        row[
            "status"
        ] = "failed"

        row[
            "error"
        ] = (
            traceback.format_exc()
        )

        row[
            "runtime_sec"
        ] = (
            time.time()
            - t0
        )

        print(
            "❌ ERREUR"
        )

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


        # Nettoyage mémoire
        for variable_name in [

            "gcamp",
            "atlas",
            "roi_mask",

            "clean_atlas",
            "atlas_6",

            "masque_sub",
            "info_masque_sub",

            "regions",
            "labels_valides",

            "ts",
            "scaled_for_curbd",

            "model",

            "pVar",
            "chi2",

            "pVar_train",
            "pVar_free",

            "chi2_train",
            "chi2_free",

            "J_final",

            "to_save",
        ]:

            globals().pop(
                variable_name,
                None,
            )

        gc.collect()


# ============================================================
# RÉSUMÉ FINAL
# ============================================================

df = pd.DataFrame(
    rows
)

print(
    "\n"
    + "=" * 100
)

print(
    "TRAIN MULTI-SOURIS TERMINÉ"
)

print(
    "=" * 100
)

done = df[
    df["status"] == "done"
].copy()

if not done.empty:

    cols = [

        "cohort",
        "month",
        "mouse",

        "n_subregions",

        "pVar_train_end",

        "pVar_free_mean",
        "pVar_free_std",

        "chi2_free_mean",

        "runtime_sec",
    ]

    print(
        done[
            cols
        ].to_string(
            index=False
        )
    )


print(
    f"\nCSV : "
    f"{results_csv}"
)

if not done.empty:
    mean_pvar = done["pVar_free_mean"].mean()
    std_pvar = done["pVar_free_mean"].std(ddof=0)
    min_pvar = done["pVar_free_mean"].min()
    max_pvar = done["pVar_free_mean"].max()

    print(
        f"\nRésumé groupe M6 : "
        f"pVar libre moyen = {mean_pvar:.4f} ± {std_pvar:.4f}"
    )

    print(
        f"Étendue pVar libre = "
        f"[{min_pvar:.4f}, {max_pvar:.4f}]"
    )


if not done.empty:
    print("\nRésumé par cohorte :")

    cohort_summary_df = (
        done
        .groupby("cohort", as_index=False)
        .agg(
            n_souris=("mouse", "count"),
            pVar_moyen=("pVar_free_mean", "mean"),
            pVar_std=("pVar_free_mean", "std"),
            pVar_min=("pVar_free_mean", "min"),
            pVar_max=("pVar_free_mean", "max"),
            runtime_moyen_sec=("runtime_sec", "mean"),
        )
    )

    print(
        cohort_summary_df.to_string(
            index=False
        )
    )

print(
    "=" * 100
)
