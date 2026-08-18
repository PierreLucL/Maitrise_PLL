### Importations ###
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.gridspec import GridSpec
from pathlib import Path

from maitrise_curbd.io import load_dataset
from maitrise_curbd.masks import (remove_thin_label_artifacts,
    reduce_atlas_to_6_regions,subdivide_mask_by_spatial_clustering, 
    build_parent_regions_dict)

from maitrise_curbd.timeseries import (extract_timeseries_du_tenseur,
    compute_dff, regress_out_global_signal, smooth_timeseries,
    smooth_timeseries,
    compute_mean_psd,
    compute_mean_autocorrelation,
    estimate_autocorrelation_timescale,
)

from maitrise_curbd.curbd import computeCURBD, trainMultiRegionRNN
from maitrise_curbd.plotting import gradient_line, plot_10_ts_with_mask_and_similarity

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
n_pixels = 50
lissage_sigma = 4
Combien_de_petites_regions = 5
nRunTrain = 500
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

# Extraction
ts = extract_timeseries_du_tenseur(
    gcamp,
    masque_sub
)

# ΔF/F
ts = compute_dff(
    ts,
    fs=12,
    window_sec=60,
    percentile=8
)

# Régression globale
ts = regress_out_global_signal(
    ts
)

# Lissage EN DERNIER
timeseries = smooth_timeseries(
    ts,
    sigma=lissage_sigma
)
### Z-score
#ts = (ts - ts.mean(axis=1, keepdims=True)) / ts.std(axis=1, keepdims=True)


#########################################################################################################
# Plotting midgame
##########################################################################################################

if plot:
    plot_10_ts_with_mask_and_similarity(ts, masque_sub, n=10, souris=souris, n_pixels=n_pixels)

#########################################################################################################
# FUCKING CURBD
##########################################################################################################

scale = np.max(timeseries)

scaled = timeseries / scale

print("Maximum :", np.max(timeseries))
print("Minimum :", np.min(timeseries))

print(
    "% > 0.999 :",
    100 * np.mean(scaled > 0.999)
)

print(
    "% < -0.999 :",
    100 * np.mean(scaled < -0.999)
)

### Training pour obtenir la matrice J

model = trainMultiRegionRNN(timeseries, dtData=1/12, # pas de temps de données (en secondes)
    dtFactor=4, # Combien de pas entre les pas de données et les pas de RNN
    tauRNN=0.3, # constante de temps du RNN (en secondes)
    nRunTrain=nRunTrain, # Nombre d'itérations d'entraînement du RNN,
    P0=1.0, # Taux d'apprentissage du RLS,
    plotStatus=True,
    regions=regions, # Dictionnaire des régions parentes
)
print(f'Terminé avec pVars = {model["pVars"][-1]} et chi2 = {model["chi2s"][-1]}')

### Calcul de CURBD

curbd_arr, curbd_labels = computeCURBD(model)

### On somme la matrice des courants pour obtenir une courbe de courant total source → cible pour chaque paire de régions parentes

n_regions = curbd_arr.shape[0]

currents = {}

for iTarget in range(n_regions):
    for iSource in range(n_regions):

        C = curbd_arr[iTarget, iSource]

        # courant total vers la région cible
        current = np.sum(C, axis=0)

        currents[(iTarget, iSource)] = current

#########################################################################################################
# Plotting final
##########################################################################################################

### Couleurs pour les régions parentes
region_colors = {
    0: "#0047AB",  # Moteur secondaire - Cobalt
    1: "#FF7F00",  # Moteur secondaire - Orange
    2: "#00A550",  # Somatosensorielle - Vert
    3: "#A020F0",  # Associative - Violet
    4: "#E60026",  # Visuelle - Rouge
    5: "#00B7EB",  # Rétrospléniale - Cyan
   
}

### Setup pour les limites des axes

all_currents = np.concatenate(
    list(currents.values())
)

max_abs = np.percentile(np.abs(all_currents), 99)

### Setup de la figure du masque coloré

mask_rgb = np.ones((*masque_sub.shape, 3))  # fond blanc

for iRegion in range(len(regions)):
    region_name = regions[iRegion, 0]
    subregion_indices = regions[iRegion, 1]

    color = to_rgb(region_colors[iRegion])

    for idx in subregion_indices:
        mask_rgb[masque_sub == idx] = color

if np.any(np.isnan(masque_sub)):
    mask_rgb[np.isnan(masque_sub)] = [1, 1, 1]


### Setup de la figure finale

fig = plt.figure(figsize=(12, 8))

outer = GridSpec(
    1, 2,
    width_ratios=[1, 5],
    wspace=0.15,
    figure=fig
)

# Colonne de gauche : masque coloré avec les régions parentes
ax_mask = fig.add_subplot(outer[0, 0])
ax_mask.imshow(mask_rgb)
ax_mask.set_title("Régions", fontsize=10)
ax_mask.axis("off")

# Matrice de droite : courants CURBD source → cible
right = outer[0, 1].subgridspec(
    n_regions,
    n_regions,
    wspace=0.08,
    hspace=0.08
)

axes = np.empty((n_regions, n_regions), dtype=object)

for iTarget in range(n_regions):
    for iSource in range(n_regions):

        ax = fig.add_subplot(right[iTarget, iSource])
        axes[iTarget, iSource] = ax

        current = currents[(iTarget, iSource)]

        source_color = region_colors[iSource]
        target_color = region_colors[iTarget]

        gradient_line(
            model["tRNN"],
            current,
            ax,
            source_color,
            target_color,
            lw=0.8 if iSource == iTarget else 0.5
        )

        ax.axhline(0, color="black", linewidth=0.4, alpha=0.25)

        ax.set_xlim(model["tRNN"][0], model["tRNN"][-1])
        ax.set_ylim(-max_abs, max_abs)

        # Colonnes = sources
        if iTarget == 0:
            ax.set_title(
                regions[iSource, 0],
                fontsize=8,
                color=source_color
            )

        # Lignes = cibles
        if iSource == 0:
            ax.set_ylabel(
                regions[iTarget, 0],
                fontsize=8,
                color=target_color
            )

        if iTarget != n_regions - 1:
            ax.set_xticklabels([])

        if iSource != 0:
            ax.set_yticklabels([])

        ax.tick_params(
            axis="both",
            labelsize=6,
            length=2
        )

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_alpha(0.5)


fig.suptitle(
    "Courants CURBD source → cible",
    fontsize=14,
    y=0.98
)

plt.show()