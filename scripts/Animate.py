from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from maitrise_curbd.timeseries import (
    compute_dff,
    extract_timeseries_du_tenseur,
    regress_out_global_signal,
    smooth_timeseries)
from pathlib import Path
import tifffile as tiff
from maitrise_curbd.io import load_dataset
from maitrise_curbd.Animation import animate_dff, animate_subregions
from maitrise_curbd.masks import (remove_thin_label_artifacts,
    reduce_atlas_to_6_regions,subdivide_mask_by_spatial_clustering, 
    build_parent_regions_dict)

# =============================================================================
# Cohortes disponibles
#
# C0 : M10[253] | M12[191,210,213,233] | M20[42,253]
# C2 : M6-16[308] | M20[304]
# C3 : M6-14[316,322] | M16-20[316]
# C5 : M6[353,361] | M8-14[353] | M18-20[359]
# C6 : M6[365,367,374] | M8-18[374]
# C7 : M6[387,396,397] | M10-12[387]
# C8 : M6[409] | M18-20[408]
# C9 : M6[410,415] | M8[410,412,415] | M10-12[410,412]
#      M14-16[410] | M18-20[415]
# =============================================================================

#########################################################################################################
# SETUP
##########################################################################################################

n_cohorte = 8
month = 6
souris = 409
n_pixels = 300
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

### On clean le masque, on extrait les pixels actifs, on retire les pixels morts du masque, et on subdivise le masque
clean_atlas = remove_thin_label_artifacts(atlas,size=5,min_fraction=0.25)

### On réduit l'atlas à 6 régions
atlas_6 = reduce_atlas_to_6_regions(
    atlas=clean_atlas,
    roi_mask=roi_mask,
)

### On subdivise le masque réduit en sous-régions
masque_sub, info_masque_sub = subdivide_mask_by_spatial_clustering(atlas_6, target_size=n_pixels)


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
ts = extract_timeseries_du_tenseur(gcamp, masque_sub)


#########################################################################################################
# Opérations sur les TS
##########################################################################################################

### Lissage
ts = smooth_timeseries(ts, sigma=lissage_sigma)

### Delta F / F comme l'article de référence rolling window
ts = compute_dff(ts)

### Regréssion du signal global
ts = regress_out_global_signal(ts)


ani = animate_subregions(
    ts=ts,
    masque_sub=masque_sub,
    interval=50,
    cmap_name="RdBu_r",
    center_zero=True)