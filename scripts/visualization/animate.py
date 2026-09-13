### Script d'animation rapide pour inspecter une souris sans partir tout le pipeline.
### C'est le mode "je veux voir si mes sous-regions bougent comme du monde".

import numpy as np
from maitrise_curbd.timeseries import (
    extract_timeseries_du_tenseur,
    regress_out_global_signal,
    smooth_timeseries,
)
from maitrise_curbd.io import load_dataset
from maitrise_curbd.animation import animate_subregions
from maitrise_curbd.masks import (
    remove_thin_label_artifacts,
    reduce_atlas_to_6_regions,
    subdivide_mask_by_spatial_clustering,
    build_parent_regions_dict,
)

# =============================================================================
# Cohortes disponibles
#
# C0 : M10[253] | M12[191,210,213,233] | M20[253]
# C2 : M8-16[308] | M20[304]
# C3 : M6-14[316,322] | M16-20[316]
# C5 : M6[353,361] | M8[353] | M10[353] | M18[359] | M20[359]
# C6 : M6[365,367,374] | M8-18[374]
# C7 : M6[387,396,397] | M10-12[387]
# C8 : M6[409] | M18-20[408]
# C9 : M6[410,415] | M8[410,412,415] | M10-12[410,412]
#      M14[410] | M18[415] | M20[415]
# =============================================================================

#########################################################################################################
### SETUP: change ici pour choisir la souris et la granularite du masque.
##########################################################################################################

n_cohorte = 7
month = 10
souris = 387
n_pixels = 50
lissage_sigma = 2
Combien_de_petites_regions = 5
nRunTrain = 100 
debug = True
plot = True

#########################################################################################################
### Preworkout: charger les donnees, nettoyer le masque, extraire les traces.
##########################################################################################################

### On sort les données
gcamp, atlas, roi_mask = load_dataset(
    cohort=n_cohorte,
    month=month,
    mouse=souris)

### On clean le masque avant de subdiviser, sinon les petites cochonneries deviennent des sous-regions.
clean_atlas = remove_thin_label_artifacts(atlas,size=5,min_fraction=0.25)

### On réduit l'atlas à 6 régions
atlas_6 = reduce_atlas_to_6_regions(
    atlas=clean_atlas,
    roi_mask=roi_mask,
)

### On subdivise le masque réduit en sous-régions
masque_sub, info_masque_sub = subdivide_mask_by_spatial_clustering(atlas_6, target_size=n_pixels)


### Building du dictionnaire de regions parentes: format partage avec CURBD.
regions = build_parent_regions_dict(info_masque_sub)

### A la chasse aux petites regions: sanity check rapide de la subdivision.
n_regions = len(np.unique(masque_sub[~np.isnan(masque_sub)]))
tailles_regions = sorted(((np.sum(masque_sub == l), l) for l in np.unique(masque_sub) if not np.isnan(l)),key=lambda x: x[0])

print('Il y a un total de {} sous-régions'.format(n_regions))
print(f"Les tailles respectives en pixels des {Combien_de_petites_regions} plus petites régions sont "+ ", ".join(
        str(int(taille))
        for taille, _ in tailles_regions[:Combien_de_petites_regions]))

### Une trace moyenne par sous-region.
ts = extract_timeseries_du_tenseur(gcamp, masque_sub)


#########################################################################################################
# Opérations sur les TS
##########################################################################################################

### Lissage: calme le bruit frame-to-frame.
#ts = smooth_timeseries(ts, sigma=lissage_sigma)

### Les fichiers GCaMP recus sont deja en DeltaF/F. On ne recalcule pas ca ici.

### Regression du signal global: enleve la composante commune.
ts = regress_out_global_signal(ts)

### Animation finale: chaque sous-region reprend sa valeur dans l'espace.
ani = animate_subregions(
    ts=ts,
    masque_sub=masque_sub,
    interval=50,
    cmap_name="RdBu_r",
    center_zero=True)
