import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import sys
import time
import traceback

from datetime import datetime
from itertools import product
from pathlib import Path

### Permet de lancer le script direct avec `python scripts/...` sans gosser avec PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

### Evite les warnings/latences Matplotlib sur les noeuds de calcul.
### Sur Narval, on veut que ca roule, pas que ca chiale pour un cache.
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.environ.get("SLURM_TMPDIR", os.environ.get("TMPDIR", "/tmp")),
)

from maitrise_curbd.curbd import computeCURBD, trainMultiRegionRNN
from maitrise_curbd.io import load_dataset
from maitrise_curbd.masks import (
    build_parent_regions_dict,
    reduce_atlas_to_6_regions,
    remove_thin_label_artifacts,
    subdivide_mask_by_spatial_clustering,
)
from maitrise_curbd.timeseries import (
    extract_timeseries_du_tenseur,
    regress_out_global_signal,
    smooth_timeseries,
)


### ============================================================================
### CONFIG SIMPLE
### ============================================================================
###
### Pour faire une nouvelle loop:
### 1. Change EXPERIMENT_NAME.
### 2. Mets les souris a analyser dans DATASETS.
### 3. Mets les parametres fixes dans BASE_PARAMS.
### 4. Mets seulement ce que tu veux varier dans SWEEP.
###
### Exemples de SWEEP:
###
###   SWEEP = {"P0": [0.1, 0.3, 1.0, 3.0]}
###   SWEEP = {"n_pixels": [50, 80, 100, 150]}
###   SWEEP = {"tauRNN": [0.083, 0.167, 0.33], "dtFactor": [2, 4]}
###   SWEEP = {"g": [0.8, 1.0, 1.5], "ampInWN": [0.001, 0.01]}
###
### Si SWEEP contient plusieurs parametres, LOOP.py teste toutes les combinaisons.
### Bref: tu changes le haut du fichier, tu lances, pis tu laisses la machine suer.

EXPERIMENT_NAME = "CURBD_signal_deja_dff_GSR_OFF_2souris"

DATASETS = [
    (9, 6, 415),
    (9, 6, 410),
]

BASE_PARAMS = {
    "seed": None,
    "segmentation_seed": 0,
    ### Les fichiers GCaMP recus sont deja en DeltaF/F. On ne refait jamais cette etape ici.
    "use_global_regression": False,
    "lissage_sigma": 4,
    "atlas_clean_size": 5,
    "atlas_clean_min_fraction": 0.25,
    "dtData": 1 / 12,
    "tauRNN": 0.33,
    "dtFactor": 2,
    "g": 0.8,
    "P0": 1.0,
    "tauWN": 0.1,
    "ampInWN": 0.01,
    "nRunTrain": 200,
    "nRunFree": 5,
    "compute_curbd": False,
}

SWEEP = {"n_pixels": [50, 80, 100, 150]}

### ============================================================================
### OUTILS
### ============================================================================

RESULTS_ROOT = Path("results")


### Garde seulement les valeurs finies. Les NaN restent dehors, ils ont assez participe.
def finite_values(x):
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


### Stats safe: retournent NaN au lieu de planter quand une liste est vide.
def safe_nanmax(x):
    x = finite_values(x)
    return float(np.max(x)) if x.size else np.nan


def safe_nanmin(x):
    x = finite_values(x)
    return float(np.min(x)) if x.size else np.nan


def safe_nanmean(x):
    x = finite_values(x)
    return float(np.mean(x)) if x.size else np.nan


def safe_nanstd(x):
    x = finite_values(x)
    return float(np.std(x)) if x.size else np.nan


def safe_last(x):
    x = finite_values(x)
    return float(x[-1]) if x.size else np.nan


### Lit une valeur a un index sans exploser si la serie est trop courte.
def safe_index(x, idx):
    x = np.asarray(x, dtype=float).ravel()

    if x.size == 0:
        return np.nan

    if idx < 0:
        idx = x.size + idx

    if idx < 0 or idx >= x.size:
        return np.nan

    value = x[idx]
    return float(value) if np.isfinite(value) else np.nan


### Supporte les modeles dict et objets. On ne sait jamais comment le futur nous niaise.
def get_model_value(model, key, default=None):
    if isinstance(model, dict):
        return model.get(key, default)

    return getattr(model, key, default)


### Transforme SWEEP en liste de configs completes. Plusieurs keys = produit cartesien.
def build_sweep_configs(base_params, sweep):
    if not sweep:
        ### Pas de sweep? Une seule config avec les parametres fixes.
        return [base_params.copy()]

    sweep_names = list(sweep)
    sweep_values = [sweep[name] for name in sweep_names]

    configs = []
    for values in product(*sweep_values):
        ### Copie la base, puis injecte les valeurs testees pour cette combinaison.
        config = base_params.copy()
        config.update(dict(zip(sweep_names, values)))
        configs.append(config)

    return configs


### Charge une config JSON externe: pratique pour Narval/SLURM sans modifier ce fichier.
def load_loop_config(config_path):
    with Path(config_path).open("r") as f:
        config = json.load(f)

    experiment_name = config.get("experiment_name", EXPERIMENT_NAME)
    datasets = [
        tuple(dataset)
        for dataset in config.get("datasets", DATASETS)
    ]
    base_params = BASE_PARAMS.copy()
    base_params.update(config.get("base_params", {}))
    sweep = config.get("sweep", SWEEP)

    return experiment_name, datasets, base_params, sweep


### Force les sorties sous results/. Comme ca, pas de pkl qui pop n'importe ou.
def resolve_output_dir(output_dir, experiment_name, timestamp):
    if output_dir is None:
        ### Cas normal: results/nom_experience/run_du_date.
        return RESULTS_ROOT / experiment_name / f"run_du_{timestamp}"

    output_dir = Path(output_dir)

    if output_dir.is_absolute():
        ### On refuse les chemins absolus pour garder le repo clean et previsible.
        raise ValueError(
            "--output-dir doit etre relatif pour garantir une sortie dans results/."
        )

    if output_dir == RESULTS_ROOT or RESULTS_ROOT in output_dir.parents:
        ### Si tu donnes deja results/quelque_chose, on respecte.
        return output_dir

    ### Sinon on prepende results/. C'est notre garde-fou anti-bordel.
    return RESULTS_ROOT / output_dir


### Charge un dataset et fait tout le preprocessing jusqu'aux timeseries pretes pour CURBD.
def prepare_timeseries(dataset, data_root, params):
    cohort, month, mouse = dataset

    ### 1. Charger le film, l'atlas et le ROI mask.
    gcamp, atlas, roi_mask = load_dataset(
        cohort=cohort,
        month=month,
        mouse=mouse,
        data_root=data_root,
    )

    ### 2. Convertir en arrays numpy standard, parce qu'apres ca tout le pipeline parle numpy.
    gcamp = np.asarray(gcamp)
    atlas = np.asarray(atlas)
    roi_mask = np.asarray(roi_mask)

    if gcamp.ndim != 3:
        ### Le film doit etre temps x hauteur x largeur. Sinon on stop tout de suite.
        raise ValueError(
            "gcamp doit etre un tableau 3D de forme (T, H, W). "
            f"Forme recue: {gcamp.shape}"
        )

    if roi_mask.shape != gcamp.shape[1:]:
        raise ValueError(
            f"roi_mask {roi_mask.shape} incompatible avec gcamp {gcamp.shape}"
        )

    ### 3. Nettoyer les petits artefacts du masque Allen avant de reduire les regions.
    clean_atlas = remove_thin_label_artifacts(
        atlas,
        size=params["atlas_clean_size"],
        min_fraction=params["atlas_clean_min_fraction"],
    )

    ### 4. Reduire vers les 6 grosses regions de travail.
    atlas_6 = reduce_atlas_to_6_regions(
        atlas=clean_atlas,
        roi_mask=roi_mask,
    )

    ### 5. Subdiviser les grosses regions en sous-regions de taille comparable.
    masque_sub, info_masque_sub = subdivide_mask_by_spatial_clustering(
        atlas_6,
        target_size=params["n_pixels"],
        random_state=params.get("segmentation_seed", 0),
    )

    ### 6. Construire le format regions attendu par CURBD.
    regions = build_parent_regions_dict(info_masque_sub)

    labels_valides = np.unique(masque_sub[np.isfinite(masque_sub)])
    n_subregions = len(labels_valides)
    n_parent_regions = len(regions)

    ### 7. Extraire les traces moyennes par sous-region.
    ts = extract_timeseries_du_tenseur(
        gcamp,
        masque_sub,
    )

    ts = np.asarray(ts, dtype=np.float32)

    if ts.ndim != 2:
        raise ValueError(
            "Les series temporelles doivent etre un tableau 2D. "
            f"Forme recue: {ts.shape}"
        )

    if params["use_global_regression"]:
        ### GSR optionnel: enleve le signal commun global si on veut.
        ts = regress_out_global_signal(ts)

    ### Lissage en dernier, parce qu'on veut smoother le signal deja nettoye.
    ts = smooth_timeseries(
        ts,
        sigma=params["lissage_sigma"],
    )

    ts = np.asarray(ts, dtype=np.float32)

    if not np.all(np.isfinite(ts)):
        ### Dernier check avant training: pas de NaN/inf dans le RNN, sinon c'est la soupe.
        n_nan = int(np.sum(np.isnan(ts)))
        n_inf = int(np.sum(np.isinf(ts)))
        raise ValueError(
            "Les series temporelles pretraitees contiennent "
            f"{n_nan} NaN et {n_inf} inf."
        )

    return {
        "ts": ts,
        "masque_sub": masque_sub,
        "info_masque_sub": info_masque_sub,
        "regions": regions,
        "n_subregions": n_subregions,
        "n_parent_regions": n_parent_regions,
        "duration_sec": ts.shape[-1] * params["dtData"],
    }


### Nomme les fichiers de sortie avec les parametres importants. Long, mais sauvagement utile.
def build_run_name(i_config, dataset, params, sweep_names):
    cohort, month, mouse = dataset

    ### Les morceaux fixes qui rendent chaque pkl identifiable dans un dossier plein.
    pieces = [
        f"config{i_config:03d}",
        f"C{cohort}",
        f"M{month}",
        f"mouse{mouse}",
    ]

    ### Les parametres qu'on veut presque toujours voir dans le nom du fichier.
    important_names = [
        "n_pixels",
        "lissage_sigma",
        "tauRNN",
        "dtFactor",
        "g",
        "ampInWN",
        "P0",
        "nRunTrain",
        "use_global_regression",
        "seed",
        "segmentation_seed",
    ]

    for name in important_names:
        if name in params and params[name] is not None:
            ### Format compact pour les floats: assez lisible, pas 45 decimales de chaos.
            value = params[name]
            if isinstance(value, float):
                pieces.append(f"{name}{value:.4g}")
            else:
                pieces.append(f"{name}{value}")

    for name in sweep_names:
        if name not in important_names:
            ### Si tu sweep un parametre custom, on l'ajoute au nom aussi.
            pieces.append(f"{name}{params[name]}")

    return "_".join(pieces) + ".pkl"


### Cree la ligne CSV de base avec toutes les colonnes attendues.
def base_row(i_config, dataset, params, save_path):
    cohort, month, mouse = dataset
    dt_rnn = params["dtData"] / params["dtFactor"]

    row = {
        ### Identite du run.
        "i_config": i_config,
        "cohort": cohort,
        "month": month,
        "mouse": mouse,
        ### Geometrie/preprocessing.
        "n_pixels": params["n_pixels"],
        "n_subregions": np.nan,
        "n_parent_regions": np.nan,
        "fps": 1 / params["dtData"],
        ### Timing RNN.
        "dtData": params["dtData"],
        "dtFactor": params["dtFactor"],
        "dtRNN": dt_rnn,
        "alpha_dt_tau": dt_rnn / params["tauRNN"],
        "lissage_sigma_frames": params["lissage_sigma"],
        "lissage_sigma_sec": params["lissage_sigma"] * params["dtData"],
        "lissage_fwhm_sec": 2.355 * params["lissage_sigma"] * params["dtData"],
        ### Hyperparametres CURBD.
        "tauRNN": params["tauRNN"],
        "g": params["g"],
        "P0": params["P0"],
        "tauWN": params["tauWN"],
        "ampInWN": params["ampInWN"],
        "nRunTrain": params["nRunTrain"],
        "nRunFree": params["nRunFree"],
        "use_global_regression": params["use_global_regression"],
        "compute_curbd": params["compute_curbd"],
        ### Diagnostics de normalisation/clipping.
        "fraction_clip_pos": np.nan,
        "fraction_clip_neg": np.nan,
        "pVar_max": np.nan,
        "pVar_finale": np.nan,
        "chi2_min": np.nan,
        "chi2_final": np.nan,
        ### Les passages free repetent le meme bruit; ils ne sont pas des essais independants.
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
        "duration_sec": np.nan,
        "runtime_sec": np.nan,
        ### Status + erreur: le CSV raconte aussi les crashs, pas juste les succes.
        "status": "started",
        "error": None,
        "save_path": str(save_path),
    }

    for name, value in params.items():
        if name not in row:
            ### Les parametres custom du sweep sont copies dans le CSV automatiquement.
            row[name] = value

    return row


### Provenance du code et de la cible utilisee par ce calcul.
def reproducibility_metadata(ts):
    ### L'empreinte des traces permet de verifier qu'on entraine exactement la meme cible.
    ts = np.ascontiguousarray(ts)
    files = [Path(__file__).resolve(), *sorted((SRC_DIR / 'maitrise_curbd').glob('*.py'))]
    versions = {}
    for package in ('numpy', 'scipy', 'pandas', 'scikit-learn', 'tifffile', 'matplotlib'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        'schema_version': 1,
        'python': platform.python_version(),
        'platform': platform.platform(),
        'packages': versions,
        'threads': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        'source_sha256': {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        'timeseries_sha256': hashlib.sha256(str((ts.shape, ts.dtype)).encode() + ts.tobytes()).hexdigest(),
    }


### Lance une config complete: preprocessing, training, metriques, sauvegarde.
def run_one_config(i_config, dataset, params, data_root, save_path):
    t0 = time.time()
    row = base_row(i_config, dataset, params, save_path)

    try:
        cohort, month, mouse = dataset

        ### Petit print de run: quand ca roule sur cluster, c'est ton fil d'Ariane.
        print("\n" + "=" * 90)
        print(f"CONFIGURATION {i_config + 1}")
        print(f"Dataset: C{cohort} M{month} souris {mouse}")
        print(
            "Parametres: "
            f"n_pixels={params['n_pixels']}, "
            f"sigma={params['lissage_sigma']}, "
            f"tauRNN={params['tauRNN']}, "
            f"dtFactor={params['dtFactor']}, "
            f"g={params['g']}, "
            f"ampInWN={params['ampInWN']}, "
            f"P0={params['P0']}"
        )
        print("=" * 90)

        ### Preprocessing complet pour ce dataset/config.
        prepared = prepare_timeseries(dataset, data_root, params)
        ts = prepared["ts"]
        regions = prepared["regions"]

        row["n_subregions"] = prepared["n_subregions"]
        row["n_parent_regions"] = prepared["n_parent_regions"]
        row["duration_sec"] = prepared["duration_sec"]

        ### Check de clipping avant CURBD: si tout tape a 0.999, le modele voit une brique.
        curbd_scale = np.max(ts)
        if not np.isfinite(curbd_scale) or curbd_scale <= 0:
            raise ValueError("Maximum invalide avant CURBD.")

        scaled_for_curbd = ts / curbd_scale
        row["fraction_clip_pos"] = float(np.mean(scaled_for_curbd > 0.999))
        row["fraction_clip_neg"] = float(np.mean(scaled_for_curbd < -0.999))

        ### Training RNN data-constrained. C'est le bout couteux, donc tout avant doit etre clean.
        model = trainMultiRegionRNN(
            ts,
            dtData=params["dtData"],
            dtFactor=params["dtFactor"],
            g=params["g"],
            tauRNN=params["tauRNN"],
            tauWN=params["tauWN"],
            ampInWN=params["ampInWN"],
            nRunTrain=params["nRunTrain"],
            nRunFree=params["nRunFree"],
            P0=params["P0"],
            regions=regions,
            seed=params.get("seed"),
            plotStatus=False,
        )

        ### On extrait les sorties du modele dans des arrays standard.
        J_final = get_model_value(model, "J")
        RNN_final = get_model_value(model, "RNN")
        Adata = get_model_value(model, "Adata")
        tData = get_model_value(model, "tData")
        tRNN = get_model_value(model, "tRNN")
        pVar = get_model_value(model, "pVars", np.array([np.nan]))
        chi2 = get_model_value(model, "chi2s", np.array([np.nan]))

        if J_final is None:
            raise KeyError("Le modele ne contient pas J.")

        J_final = np.asarray(J_final, dtype=np.float32)
        RNN_final = None if RNN_final is None else np.asarray(RNN_final, dtype=np.float32)
        Adata = None if Adata is None else np.asarray(Adata, dtype=np.float32)
        tData = None if tData is None else np.asarray(tData, dtype=np.float32)
        tRNN = None if tRNN is None else np.asarray(tRNN, dtype=np.float32)
        pVar = np.asarray(pVar, dtype=float)
        chi2 = np.asarray(chi2, dtype=float)

        ### Separation train/free: on veut savoir si le modele apprend ET s'il tient sans apprendre.
        pVar_train = pVar[:params["nRunTrain"]]
        pVar_free = pVar[
            params["nRunTrain"]:
            params["nRunTrain"] + params["nRunFree"]
        ]

        chi2_train = chi2[:params["nRunTrain"]]
        chi2_free = chi2[
            params["nRunTrain"]:
            params["nRunTrain"] + params["nRunFree"]
        ]

        ### Remplissage des metriques CSV. Si une serie est vide, les safe_* mettent NaN.
        row["pVar_max"] = safe_nanmax(pVar)
        row["pVar_finale"] = safe_last(pVar)
        row["chi2_min"] = safe_nanmin(chi2)
        row["chi2_final"] = safe_last(chi2)
        row["pVar_max_train"] = safe_nanmax(pVar_train)
        row["pVar_train_end"] = safe_index(pVar, params["nRunTrain"] - 1)
        row["pVar_free_mean"] = safe_nanmean(pVar_free)
        row["pVar_free_std"] = safe_nanstd(pVar_free)
        row["pVar_free_min"] = safe_nanmin(pVar_free)
        row["pVar_free_final"] = safe_last(pVar_free)
        row["chi2_min_train"] = safe_nanmin(chi2_train)
        row["chi2_train_end"] = safe_index(chi2, params["nRunTrain"] - 1)
        row["chi2_free_mean"] = safe_nanmean(chi2_free)
        row["chi2_free_std"] = safe_nanstd(chi2_free)
        row["chi2_free_min"] = safe_nanmin(chi2_free)
        row["chi2_free_final"] = safe_last(chi2_free)

        ### Optionnel: computeCURBD est plus lourd, donc on le garde toggle-able.
        curbd_arr = None
        curbd_labels = None
        if params["compute_curbd"]:
            curbd_arr, curbd_labels = computeCURBD(model)

        row["runtime_sec"] = time.time() - t0
        row["status"] = "done"

        ### Le pkl garde assez d'info pour analyser plus tard sans refaire l'entrainement.
        to_save = {
            ### Pleine precision pour rejouer le modele; les anciens champs restent en float32.
            "J0": get_model_value(model, "J0"),
            "inputWN": get_model_value(model, "inputWN"),
            "J_final_full_precision": get_model_value(model, "J"),
            "initial_state": get_model_value(model, "initial_state"),
            "activity_scale": get_model_value(model, "activity_scale"),
            "model_parameters": get_model_value(model, "params"),
            "iTarget": get_model_value(model, "iTarget"),
            "masque_sub": prepared["masque_sub"],
            "info_masque_sub": prepared["info_masque_sub"],
            "reproducibility": reproducibility_metadata(ts),
            "J_final": J_final,
            "RNN_final": RNN_final,
            "Adata": Adata,
            "tData": tData,
            "tRNN": tRNN,
            "pVar": pVar,
            "chi2": chi2,
            "regions": regions,
            "curbd_arr": curbd_arr,
            "curbd_labels": curbd_labels,
            "parameters": {
                "cohort": cohort,
                "month": month,
                "mouse": mouse,
                **params,
                "n_subregions": row["n_subregions"],
                "n_parent_regions": row["n_parent_regions"],
                "fraction_clip_pos": row["fraction_clip_pos"],
                "fraction_clip_neg": row["fraction_clip_neg"],
            },
            "row": row.copy(),
        }

        ### Le fichier final n'apparait qu'une fois la sauvegarde terminee.
        temporary_path = save_path.with_suffix(save_path.suffix + '.tmp')
        try:
            with temporary_path.open("wb") as f:
                pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            temporary_path.replace(save_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        print(f"Sauvegarde: {save_path}")
        print(f"pVar train end: {row['pVar_train_end']:.4f}")
        print(f"pVar free mean: {row['pVar_free_mean']:.4f}")
        print(f"chi2 free mean: {row['chi2_free_mean']:.4f}")
        print(f"runtime: {row['runtime_sec']:.1f} s")

    except Exception:
        ### On capture l'erreur dans le CSV au lieu de perdre toute la nuit de jobs.
        row["status"] = "failed"
        row["error"] = traceback.format_exc()
        row["runtime_sec"] = time.time() - t0
        print("ERREUR")
        print(row["error"])

    finally:
        ### Petit menage memoire entre les configs. Sur Narval, chaque Go compte.
        gc.collect()

    return row


### Point d'entree CLI: parse les args, cree les jobs, lance la boucle.
def main():
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%Hh%M")

    ### Arguments minimalistes: data-dir, output-dir, dry-run. Pas besoin d'un cockpit d'avion.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Fichier JSON avec experiment_name, datasets, base_params et sweep.",
    )
    parser.add_argument(
        "--job-index",
        type=int,
        default=None,
        help="Index unique a lancer dans la grille. Par defaut: tous les jobs.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Racine des donnees. Si absent, utilise MAITRISE_DATA_DIR puis data/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Chemin relatif. Il sera toujours place dans results/.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    experiment_name, datasets, base_params, sweep = (
        load_loop_config(args.config)
        if args.config is not None
        else (EXPERIMENT_NAME, DATASETS, BASE_PARAMS, SWEEP)
    )

    ### Toujours sous results/, meme si tu donnes juste un petit nom de run.
    save_dir = resolve_output_dir(args.output_dir, experiment_name, timestamp)
    save_dir.mkdir(parents=True, exist_ok=True)

    ### Expansion du sweep: DATASETS x CONFIGS = jobs reels a lancer.
    sweep_names = list(sweep)
    configs = build_sweep_configs(base_params, sweep)
    jobs = [
        (dataset, config)
        for dataset in datasets
        for config in configs
    ]

    if args.job_index is None:
        env_job_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_job_index is not None:
            args.job_index = int(env_job_index)

    if args.job_index is not None:
        if args.job_index < 0 or args.job_index >= len(jobs):
            raise ValueError(
                f"--job-index {args.job_index} invalide pour {len(jobs)} jobs."
            )
        jobs = [jobs[args.job_index]]

    results_csv = (
        save_dir / f"loop_summary_job_{args.job_index:04d}.csv"
        if args.job_index is not None
        else save_dir / "loop_summary.csv"
    )

    print("\n" + "=" * 90)
    print("LOOP CURBD GENERAL")
    print("=" * 90)
    print(f"Experiment: {experiment_name}")
    print(f"Datasets: {len(datasets)}")
    print(f"Configurations par dataset: {len(configs)}")
    print(f"Total runs lances ici: {len(jobs)}")
    print(f"Total grille complete: {len(datasets) * len(configs)}")
    print(f"Job index: {args.job_index if args.job_index is not None else 'tous'}")
    print(f"Data dir: {args.data_dir}")
    print(f"Results dir: {save_dir}")
    print(f"Sweep: {sweep if sweep else 'aucun'}")

    if args.dry_run:
        ### Dry-run: montre ce qui serait lance/sauve, sans toucher aux donnees.
        for i_config, (dataset, params) in enumerate(jobs):
            job_i = args.job_index if args.job_index is not None else i_config
            run_name = build_run_name(job_i, dataset, params, sweep_names)
            print(f"{job_i:03d}: dataset={dataset} -> {save_dir / run_name}")
        return

    rows = []
    if results_csv.exists():
        ### En mode array, plusieurs jobs ecrivent dans le meme CSV: on append ce qui existe deja.
        rows = pd.read_csv(results_csv).to_dict("records")

    for local_i, (dataset, params) in enumerate(jobs):
        i_config = args.job_index if args.job_index is not None else local_i
        ### Chaque job recoit son propre pkl et une ligne dans le CSV resume.
        run_name = build_run_name(i_config, dataset, params, sweep_names)
        save_path = save_dir / run_name

        row = run_one_config(
            i_config=i_config,
            dataset=dataset,
            params=params,
            data_root=args.data_dir,
            save_path=save_path,
        )

        ### On ecrit le CSV apres chaque config pour garder les resultats meme si ca plante apres.
        rows.append(row.copy())
        pd.DataFrame(rows).to_csv(results_csv, index=False)
        print(f"CSV resume mis a jour: {results_csv}")

    print("\n" + "=" * 90)
    print("LOOP TERMINEE")
    print(f"Resultats: {save_dir}")
    print(f"CSV resume: {results_csv}")
    print("=" * 90)


if __name__ == "__main__":
    main()
