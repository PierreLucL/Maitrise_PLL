#### Importations et fonctions pour l'analyse des données GCaMP ####

import h5py

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
