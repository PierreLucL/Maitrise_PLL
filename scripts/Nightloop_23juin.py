import gc
import pickle
import time
import traceback
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.maitrise_curbd.io import load_gcamp
from src.maitrise_curbd.masks import (
    build_parent_regions_dict,
    clean_region_mask,
    extract_nonzero_pixels,
    remove_dead_pixels_from_region_mask,
    subdivide_mask_by_spatial_clustering,
)
from src.maitrise_curbd.timeseries import (
    compute_dff,
    extract_timeseries_du_tenseur,
    regress_out_global_signal,
    smooth_timeseries,
)
from src.maitrise_curbd.curbd import computeCURBD, trainMultiRegionRNN


# ============================================================
# PARAMÈTRES
# ============================================================

souris = ['M387-6', 'M396-6', 'M410-6', 'M412-8']

idx_souris_list = [0, 1, 2, 3]
n_pixels_list = [50,30]

lissage_sigma = 2
use_dff = False
use_global_regression = True

nRunTrain = 1000
debug = False

dtData = 0.33
dtFactor = 4
tauRNN = 0.33
P0 = 1.0

base_path = Path(
    "/Users/pierre-luclarouche/Desktop/École/Maîtrise/Maitrise_PLL/Coding CURBD 2026"
)

save_dir = Path("night_run_J_CURBD_36courbes")
save_dir.mkdir(exist_ok=True)

results_csv = save_dir / "night_run_summary.csv"


# ============================================================
# FONCTIONS SAFE POUR MÉTRIQUES
# ============================================================

def safe_nanmax(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(np.isnan(x)):
        return np.nan
    return float(np.nanmax(x))


def safe_nanmin(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(np.isnan(x)):
        return np.nan
    return float(np.nanmin(x))


def safe_last(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    return float(x[-1])


# ============================================================
# CONFIGS
# ============================================================

configs = list(product(idx_souris_list, n_pixels_list))
rows = []


# ============================================================
# LOOP PRINCIPALE
# ============================================================

for i_config, (Idx_souris, n_pixels) in enumerate(configs):

    t0 = time.time()

    save_path = save_dir / (
        f"config{i_config}_"
        f"{souris[Idx_souris]}_"
        f"pix{n_pixels}_"
        f"sigma{lissage_sigma}_"
        f"dff{use_dff}_"
        f"globalreg{use_global_regression}_"
        f"nRunTrain{nRunTrain}.pkl"
    )

    row = {
        "i_config": i_config,
        "Idx_souris": Idx_souris,
        "souris": souris[Idx_souris],
        "n_pixels": n_pixels,
        "n_regions": np.nan,
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
        print(f"CONFIG {i_config + 1}/{len(configs)}")
        print(f"Souris : {souris[Idx_souris]} | n_pixels : {n_pixels}")
        print("=" * 90)

        dataset, masque_init = load_gcamp(
            base_path / f"{souris[Idx_souris]}_v4_mvmt.h5"
        )

        clean_masque_init = clean_region_mask(masque_init)

        pixels_actifs, masque_mort = extract_nonzero_pixels(
            dataset,
            debug=debug
        )

        masque_init_actif = remove_dead_pixels_from_region_mask(
            clean_masque_init,
            masque_mort
        )

        masque_sub, info_masque_sub = subdivide_mask_by_spatial_clustering(
            masque_init_actif,
            target_size=n_pixels
        )

        regions = build_parent_regions_dict(info_masque_sub)

        ts = extract_timeseries_du_tenseur(
            dataset,
            masque_sub
        )

        ts = smooth_timeseries(ts, sigma=lissage_sigma)

        if use_dff:
            ts = compute_dff(ts)

        if use_global_regression:
            ts = regress_out_global_signal(ts)

        ts = np.asarray(ts, dtype=np.float32)

        model = trainMultiRegionRNN(
            ts,
            dtData=dtData,
            dtFactor=dtFactor,
            tauRNN=tauRNN,
            nRunTrain=nRunTrain,
            P0=P0,
            regions=regions,
            plotStatus=False,
        )

        curbd_arr, curbd_labels = computeCURBD(model)

        n_regions = curbd_arr.shape[0]
        row["n_regions"] = n_regions

        currents_curves = {}

        for iTarget in range(n_regions):
            for iSource in range(n_regions):

                C = curbd_arr[iTarget, iSource]

                current_curve = np.sum(C, axis=0).astype(np.float32)

                currents_curves[(iTarget, iSource)] = current_curve

        if isinstance(model, dict):
            J_final = model["J"]
            pVar = np.asarray(model.get("pVars", np.nan))
            chi2 = np.asarray(model.get("chi2s", np.nan))
        else:
            J_final = model.J
            pVar = np.asarray(getattr(model, "pVars", np.nan))
            chi2 = np.asarray(getattr(model, "chi2s", np.nan))

        J_final = np.asarray(J_final, dtype=np.float32)

        row["pVar_max"] = safe_nanmax(pVar)
        row["pVar_finale"] = safe_last(pVar)

        row["chi2_min"] = safe_nanmin(chi2)
        row["chi2_final"] = safe_last(chi2)

        to_save = {
    "J_final": J_final,
    "currents_curves": currents_curves,
    "curbd_labels": curbd_labels,
    "regions": regions,
    "info_masque_sub": info_masque_sub,
    "masque_sub": masque_sub.astype(np.float32),
    "tRNN": np.asarray(model["tRNN"], dtype=np.float32),
    "row": row,
}

        with open(save_path, "wb") as f:
            pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

        row["status"] = "done"
        row["runtime_sec"] = time.time() - t0

        print(f"✅ Sauvegarde réussie : {save_path}")
        print(f"n_regions = {n_regions}")
        print(f"Nombre de courbes sauvegardées = {len(currents_curves)}")
        print(f"pVar max = {row['pVar_max']:.4f}")
        print(f"pVar finale = {row['pVar_finale']:.4f}")
        print(f"chi2 min = {row['chi2_min']:.4f}")
        print(f"chi2 final = {row['chi2_final']:.4f}")
        print(f"runtime = {row['runtime_sec']:.1f} s")

    except Exception:

        row["status"] = "failed"
        row["error"] = traceback.format_exc()
        row["runtime_sec"] = time.time() - t0

        print("❌ ERREUR")
        print(row["error"])

    finally:

        rows.append(row)
        pd.DataFrame(rows).to_csv(results_csv, index=False)

        print(f"CSV résumé mis à jour : {results_csv}")

        for var in [
            "dataset",
            "masque_init",
            "clean_masque_init",
            "pixels_actifs",
            "masque_mort",
            "masque_init_actif",
            "masque_sub",
            "info_masque_sub",
            "regions",
            "ts",
            "model",
            "curbd_arr",
            "curbd_labels",
            "currents_curves",
            "J_final",
            "pVar",
            "chi2",
            "to_save",
        ]:
            if var in locals():
                del locals()[var]

        gc.collect()


print("\n" + "=" * 90)
print("NIGHT RUN TERMINÉE")
print(f"Résumé CSV : {results_csv}")
print("=" * 90)