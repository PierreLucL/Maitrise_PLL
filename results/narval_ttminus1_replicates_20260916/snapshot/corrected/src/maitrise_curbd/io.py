### Ici on garde tout le stock d'entree/sortie au meme endroit.
### Le but: plus jamais chercher un chemin de dataset cache dans un script random.

import os
import h5py
from pathlib import Path
import numpy as np
import tifffile

### Petite map maison des datasets disponibles.
### C'est volontairement explicite: pas sexy, mais ben pratique quand on oublie quelle souris existe.
DATASETS = {
    0: {
        10: [253],
        12: [191, 210, 213, 233],
        20: [253],
    },

    2: {
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
        18: [415],
        20: [415],
    },
}

DATA_DIR_ENV_VAR = "MAITRISE_DATA_DIR"
DEFAULT_DATA_ROOT = Path("data")


def get_data_root(data_root: str | Path | None = None) -> Path:
    """
    Retourne la racine des datasets.

    Priorité :
    1. argument explicite ``data_root``
    2. variable d'environnement ``MAITRISE_DATA_DIR``
    3. dossier local ``data``

    Sur Narval, configure par exemple :
    ``export MAITRISE_DATA_DIR=/scratch/pllar11/Datasets``
    """
    ### Si le script donne un chemin direct, on respecte ca. C'est le boss final.
    if data_root is not None:
        return Path(data_root).expanduser()

    ### Sinon on check l'environnement, parfait pour Narval/SLURM sans gosser le code.
    env_data_root = os.environ.get(DATA_DIR_ENV_VAR)
    if env_data_root:
        return Path(env_data_root).expanduser()

    ### Dernier fallback local: pratique pour tester sans setup de cluster.
    return DEFAULT_DATA_ROOT


### Transforme (cohorte, mois, souris) en chemin concret vers le dossier de donnees.
def get_dataset_folder(
    cohort: int,
    month: int,
    mouse: int,
    data_root: str | Path | None = None,
) -> Path:
    """
    Retourne le dossier contenant les données d'une souris.

    Exemple :
    data/C0_M10/Data/RS_M253
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

    ### Convention de dossiers: une racine, puis Cx_My/Data/RS_Mz. Simple, clean, pas de drama.
    return (
        get_data_root(data_root)
        / f"C{cohort}_M{month}"
        / "Data"
        / f"RS_M{mouse}"
    )


### Loader principal pour les trois fichiers qui reviennent tout le temps dans le pipeline.
def load_dataset(
    cohort: int,
    month: int,
    mouse: int,
    data_root: str | Path | None = None,
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
    folder = get_dataset_folder(cohort, month, mouse, data_root=data_root)

    ### Les noms attendus sont standardises ici pour eviter les mini-variantes partout.
    gcamp_path = folder / "GCaMP.tif"
    atlas_path = folder / "atlas.npy"
    roi_mask_path = folder / "roi_mask.tif"

    ### On fail vite si le dataset est incomplet. Mieux vaut planter ici que 30 minutes plus tard.
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

    ### Lecture des gros fichiers: tifffile pour l'imagerie, numpy pour l'atlas.
    gcamp = tifffile.imread(gcamp_path)
    atlas = np.load(atlas_path)
    roi_mask = tifffile.imread(roi_mask_path)

    return gcamp, atlas, roi_mask

### Petit loader H5 historique: garde ca ici pour ne pas melanger les formats ailleurs.
def load_gcamp(h5_path):
    """
    Ouvre le fichier H5 et retourne le dataset GCaMP en numpy array.

    Args:
        h5_path: chemin vers le fichier .h5

    Retourne:
        data_gcamp: array (T, H, W) float32, (1440, 238, 261)
        mask_registration: array (H, W) int32, (238, 261)
    """
    ### On ouvre le fichier juste le temps de sortir les arrays, puis on le referme proprement.
    with h5py.File(h5_path, "r") as f:
        data_gcamp = f["data/3d/GCaMP"][:]
        mask_registration = f["registration/atlas"][:]
    return data_gcamp, mask_registration
