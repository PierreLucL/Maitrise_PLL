#### Importations et fonctions pour l'analyse des données GCaMP ####

import h5py
from pathlib import Path
import numpy as np
import tifffile

#### LOADER LE FICHIER TIF ####
#########################################################################################################
# DATASETS
##########################################################################################################


DATASETS = {
    0: {
        10: [253],
        12: [191, 210, 213, 233],
        20: [42, 253],
    },

    2: {
        6:  [308],
        8:  [308],
        10: [308],
        12: [308],
        14: [308],
        16: [308],
        20: [304],
    },

    3: {
        6:  [316, 322],
        8:  [316, 322],
        10: [316, 322],
        12: [316, 322],
        14: [316, 322],
        16: [316],
        18: [316],
        20: [316],
    },

    5: {
        6:  [353, 361],
        8:  [353],
        10: [353],
        12: [353],
        14: [353],
        18: [359],
        20: [359],
    },

    6: {
        6:  [365, 367, 374],
        8:  [374],
        10: [374],
        16: [374],
        18: [374],
    },

    7: {
        6:  [387, 396, 397],
        10: [387],
        12: [387],
    },

    8: {
        6:  [409],
        18: [408],
        20: [408],
    },

    9: {
        6:  [410, 415],
        8:  [410, 412, 415],
        10: [410, 412],
        12: [410, 412],
        14: [410],
        16: [410],
        18: [415],
        20: [415],
    },
}

DATA_ROOT = Path("/Volumes/Toute ma vie/Datasets")


def get_dataset_folder(cohort: int, month: int, mouse: int) -> Path:
    """
    Retourne le dossier contenant les données d'une souris.

    Exemple :
    /Volumes/Toute ma vie/Datasets/C0_M10/Data/RS_M253
    """
    if cohort not in DATASETS:
        raise ValueError(f"Cohorte C{cohort} inconnue.")

    if month not in DATASETS[cohort]:
        raise ValueError(
            f"Aucune donnée pour C{cohort}_M{month}. "
            f"Mois disponibles : {sorted(DATASETS[cohort])}"
        )

    if mouse not in DATASETS[cohort][month]:
        raise ValueError(
            f"La souris {mouse} n'est pas disponible pour C{cohort}_M{month}. "
            f"Souris disponibles : {DATASETS[cohort][month]}"
        )

    return (
        DATA_ROOT
        / f"C{cohort}_M{month}"
        / "Data"
        / f"RS_M{mouse}"
    )


def load_dataset(
    cohort: int,
    month: int,
    mouse: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge les trois fichiers principaux d'un dataset.

    Returns
    -------
    gcamp : np.ndarray
        Série temporelle GCaMP, shape attendue : (T, M, N)

    atlas : np.ndarray
        Atlas cortical, shape attendue : (M, N)

    roi_mask : np.ndarray
        Masque ROI, shape attendue : (M, N)
    """
    folder = get_dataset_folder(cohort, month, mouse)

    gcamp_path = folder / "GCaMP.tif"
    atlas_path = folder / "atlas.npy"
    roi_mask_path = folder / "roi_mask.tif"

    missing_files = [
        path.name
        for path in (gcamp_path, atlas_path, roi_mask_path)
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Fichiers manquants dans :\n{folder}\n"
            f"Fichiers manquants : {missing_files}"
        )

    gcamp = tifffile.imread(gcamp_path)
    atlas = np.load(atlas_path)
    roi_mask = tifffile.imread(roi_mask_path)

    return gcamp, atlas, roi_mask

#### LOADER LE FICHIER H5 ####

def load_gcamp(h5_path):
    """
    Ouvre le fichier H5 et retourne le dataset GCaMP en numpy array.

    Args:
        h5_path: chemin vers le fichier .h5

    Retourne:
        data_gcamp: array (T, H, W) float32, (1440, 238, 261)
        mask_registration: array (H, W) int32, (238, 261)
    """
    with h5py.File(h5_path, "r") as f:
        data_gcamp = f["data/3d/GCaMP"][:]
        mask_registration = f["registration/atlas"][:]
    return data_gcamp, mask_registration
