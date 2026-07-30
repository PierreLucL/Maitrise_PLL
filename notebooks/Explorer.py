### Importations ###
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.gridspec import GridSpec
from pathlib import Path
import tifffile as tiff
from maitrise_curbd.io import load_dataset
from maitrise_curbd.masks import remove_thin_label_artifacts,clean_reduced_atlas, reduce_atlas_to_6_regions, subdivide_mask_by_spatial_clustering, build_parent_regions_dict, clean_region_mask

#########################################################################################################
# SETUP
##########################################################################################################

n_cohorte = 0
month = 10
souris = 253
n_pixels = 700
lissage_sigma = 2
Combien_de_petites_regions = 5
nRunTrain = 100 
debug = True
plot = True

#########################################################################################################
# Preworkout
##########################################################################################################

### On sort les données
gcamp, atlas, roi_mask = load_dataset(
    cohort=n_cohorte,
    month=month,
    mouse=souris)

print("GCaMP :", gcamp.shape, gcamp.dtype)
print("Atlas :", atlas.shape, atlas.dtype)
print("ROI mask :", roi_mask.shape, roi_mask.dtype)


### On clean le masque, on extrait les pixels actifs, on retire les pixels morts du masque, et on subdivise le masque
clean_atlas = remove_thin_label_artifacts(atlas,size=5,min_fraction=0.25)

atlas_6 = reduce_atlas_to_6_regions(
    atlas=clean_atlas,
    roi_mask=roi_mask,
)
plt.imshow(atlas_6)
plt.show()

print(np.unique(atlas_6, return_counts=True))

masque_sub, info_masque_sub = subdivide_mask_by_spatial_clustering(atlas_6, target_size=n_pixels)

plt.imshow(masque_sub)
plt.show()

### Building du dictionnaire de régions parentes
regions = build_parent_regions_dict(info_masque_sub)

# À la chasse aux petites régions
n_regions = len(np.unique(masque_sub[~np.isnan(masque_sub)]))
tailles_regions = sorted(((np.sum(masque_sub == l), l) for l in np.unique(masque_sub) if not np.isnan(l)),key=lambda x: x[0])

print('Il y a un total de {} sous-régions'.format(n_regions))
print(f"Les tailles respectives en pixels des {Combien_de_petites_regions} plus petites régions sont "+ ", ".join(
        str(int(taille))
        for taille, _ in tailles_regions[:Combien_de_petites_regions]))

### Get these TS
ts = extract_timeseries_du_tenseur(dataset, masque_sub)

