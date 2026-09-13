### Importations ###
from Pipeline import *
from curbd import *
from sklearn.decomposition import PCA
import pylab
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgb
import h5py
import numpy as np
import pandas as pd
import traceback
import time
from pathlib import Path
from itertools import product

#########################################################################################################
# PARAMÈTRES À TESTER
#########################################################################################################

souris = ['M387-6', 'M396-6', 'M410-6', 'M412-8']

idx_souris_list = [0, 1, 2, 3]
n_pixels_list = [30, 50]
lissage_sigma_list = [2]
use_dff_list = [False]
use_global_regression_list = [True]

nRunTrain = 100
debug = False
plot = False

base_path = Path("/Users/pierre-luclarouche/Desktop/École/Maîtrise/Maitrise_PLL/Coding CURBD 2026")

results_path = Path("night_run_results.csv")
errors_path = Path("night_run_errors.txt")
models_dir = Path("night_run_models")
models_dir.mkdir(exist_ok=True)


#########################################################################################################
# SAUVEGARDE
#########################################################################################################

def save_row(row, path=results_path):
    df_row = pd.DataFrame([row])
    df_row.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False
    )


def config_key(row):
    return (
        row["souris"],
        row["n_pixels"],
        row["lissage_sigma"],
        row["use_dff"],
        row["use_global_regression"],
        row["nRunTrain"]
    )


#########################################################################################################
# REPRISE AUTOMATIQUE SI LE SCRIPT EST RELANCÉ
#########################################################################################################

if results_path.exists():
    old = pd.read_csv(results_path)

    done_configs = set(
        zip(
            old["souris"],
            old["n_pixels"],
            old["lissage_sigma"],
            old["use_dff"],
            old["use_global_regression"],
            old["nRunTrain"]
        )
    )
else:
    done_configs = set()


#########################################################################################################
# GRILLE DE SIMULATIONS
#########################################################################################################

configs = list(product(
    idx_souris_list,
    n_pixels_list,
    lissage_sigma_list,
    use_dff_list,
    use_global_regression_list
))

print(f"Nombre total de simulations prévues : {len(configs)}")


#########################################################################################################
# BOUCLE PRINCIPALE
#########################################################################################################

for i, (Idx_souris, n_pixels, lissage_sigma, use_dff, use_global_regression) in enumerate(configs):

    mouse_id = souris[Idx_souris]

    row = {
        "i_config": i,
        "souris": mouse_id,
        "Idx_souris": Idx_souris,
        "n_pixels": n_pixels,
        "lissage_sigma": lissage_sigma,
        "use_dff": use_dff,
        "use_global_regression": use_global_regression,
        "nRunTrain": nRunTrain,

        "n_regions": np.nan,
        "min_region_size": np.nan,
        "median_region_size": np.nan,
        "max_region_size": np.nan,

        "pVar_final": np.nan,
        "chi2_final": np.nan,
        "pVar_max": np.nan,
        "chi2_min": np.nan,

        "runtime_sec": np.nan,
        "status": "started",
        "error": ""
    }

    key = config_key(row)

    if key in done_configs:
        print(f"Déjà fait, skip : {key}")
        continue

    t0 = time.time()

    try:
        print(
            f"\n[{i+1}/{len(configs)}] "
            f"{mouse_id} | n_pixels={n_pixels} | sigma={lissage_sigma} | "
            f"dff={use_dff} | global_reg={use_global_regression}",
            flush=True
        )

        #################################################################################################
        # CHARGEMENT DES DONNÉES
        #################################################################################################

        file_path = base_path / f"{mouse_id}_v4_mvmt.h5"

        with h5py.File(file_path, "r") as f:
            infos_animal = dict(f["data"].attrs)

        dataset, masque_init = load_gcamp(file_path)

        #################################################################################################
        # MASQUE ET SOUS-RÉGIONS
        #################################################################################################

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

        labels_sub = np.unique(masque_sub[~np.isnan(masque_sub)])

        tailles_regions = np.array([
            np.sum(masque_sub == l)
            for l in labels_sub
        ])

        row["n_regions"] = len(labels_sub)
        row["min_region_size"] = int(np.min(tailles_regions))
        row["median_region_size"] = float(np.median(tailles_regions))
        row["max_region_size"] = int(np.max(tailles_regions))

        #################################################################################################
        # EXTRACTION TS
        #################################################################################################

        ts = extract_timeseries_du_tenseur(dataset, masque_sub)

        #################################################################################################
        # OPÉRATIONS SUR LES TS
        #################################################################################################

        if lissage_sigma > 0:
            ts = smooth_timeseries(ts, sigma=lissage_sigma)

        if use_dff:
            ts = compute_dff(ts)

        if use_global_regression:
            ts = regress_out_global_signal(ts)

        # Sécurité numérique
        if np.isnan(ts).any():
            raise ValueError("NaN détecté dans ts avant TrainRNN")

        if np.isinf(ts).any():
            raise ValueError("Inf détecté dans ts avant TrainRNN")

        if np.any(np.std(ts, axis=1) == 0):
            raise ValueError("Au moins une sous-région a une variance nulle")

        #################################################################################################
        # CURBD / RNN
        #################################################################################################

        model = trainMultiRegionRNN(
            ts,
            dtData=0.33,
            dtFactor=4,
            tauRNN=0.33,
            nRunTrain=nRunTrain,
            P0=1.0,
            regions=regions,plotStatus=False
        )

        pVars = np.array(model["pVars"])
        chi2s = np.array(model["chi2s"])

        row["pVar_final"] = float(pVars[-1])
        row["chi2_final"] = float(chi2s[-1])
        row["pVar_max"] = float(np.nanmax(pVars))
        row["chi2_min"] = float(np.nanmin(chi2s))

        save_name = (
            f"{mouse_id}"
            f"_npix-{n_pixels}"
            f"_sigma-{lissage_sigma}"
            f"_dff-{use_dff}"
            f"_greg-{use_global_regression}"
            f"_nrun-{nRunTrain}.pkl"
        )

        save_path = models_dir / save_name

        row["status"] = "success"

    except KeyboardInterrupt:
        row["status"] = "interrupted"
        row["error"] = "KeyboardInterrupt"
        row["runtime_sec"] = time.time() - t0
        save_row(row)

        print("\nInterruption manuelle sauvegardée. Tu n'as pas perdu les runs précédents.")
        break

    except Exception as e:
        row["status"] = "failed"
        row["error"] = str(e)

        with open(errors_path, "a") as f:
            f.write("\n\n" + "=" * 80 + "\n")
            f.write(str(row) + "\n")
            f.write(traceback.format_exc())

    finally:
        row["runtime_sec"] = time.time() - t0

        if row["status"] != "interrupted":
            save_row(row)

        print(
            f"Résultat : {row['status']} | "
            f"pVar={row['pVar_final']} | "
            f"chi2={row['chi2_final']} | "
            f"temps={row['runtime_sec']:.1f}s",
            flush=True
        )