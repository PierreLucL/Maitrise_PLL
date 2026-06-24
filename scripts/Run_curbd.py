### Importations ###
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.gridspec import GridSpec

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
from src.maitrise_curbd.plotting import gradient_line, plot_10_ts_with_mask_and_similarity

### Liste des souris disponibles ###
souris = ['M387-6', 'M396-6', 'M410-6', 'M412-8']

#########################################################################################################
# SETUP
##########################################################################################################


Idx_souris = 2
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
with h5py.File(f"/Users/pierre-luclarouche/Desktop/École/Maîtrise/Maitrise_PLL/Coding CURBD 2026/{souris[Idx_souris]}_v4_mvmt.h5", "r") as f:
   infos_animal = dict(f["data"].attrs)

### On affiche les infos de l'animal
print(
    f"""
Informations sur l'animal
-------------------------
Indice souris : {souris[Idx_souris]}  
GCaMP         : {bool(infos_animal['GCaMP'])}
Âge           : {int(infos_animal['age'])} mois
Cohorte       : {int(infos_animal['cohort'])}
Sexe          : {infos_animal['sex']}
Frames        : {infos_animal['monitoring_frame_range']}
"""
)

### On clean le masque, on extrait les pixels actifs, on retire les pixels morts du masque, et on subdivise le masque
dataset, masque_init = load_gcamp(f"/Users/pierre-luclarouche/Desktop/École/Maîtrise/Maitrise_PLL/Coding CURBD 2026/{souris[Idx_souris]}_v4_mvmt.h5")
clean_masque_init = clean_region_mask(masque_init)
pixels_actifs, masque_mort = extract_nonzero_pixels(dataset, debug=debug)
masque_init_actif = remove_dead_pixels_from_region_mask(clean_masque_init, masque_mort)
masque_sub, info_masque_sub = subdivide_mask_by_spatial_clustering(masque_init_actif, target_size=n_pixels)

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


#########################################################################################################
# Opérations sur les TS
##########################################################################################################

### Lissage
ts = smooth_timeseries(ts, sigma=lissage_sigma)

### Delta F / F comme l'article de référence rolling window
ts = compute_dff(ts)

### Regréssion du signal global
ts = regress_out_global_signal(ts)

### Z-score
#ts = (ts - ts.mean(axis=1, keepdims=True)) / ts.std(axis=1, keepdims=True)


#########################################################################################################
# Plotting midgame
##########################################################################################################

if plot:
    plot_10_ts_with_mask_and_similarity(ts, masque_sub, n=10, souris=souris[Idx_souris], n_pixels=n_pixels)

#########################################################################################################
# FUCKING CURBD
##########################################################################################################

### Training pour obtenir la matrice J

model = trainMultiRegionRNN(ts, dtData=0.33, # pas de temps de données (en secondes)
    dtFactor=4, # Combien de pas entre les pas de données et les pas de RNN
    tauRNN=0.33, # constante de temps du RNN (en secondes)
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
            lw= 3 if iSource == iTarget else 2
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