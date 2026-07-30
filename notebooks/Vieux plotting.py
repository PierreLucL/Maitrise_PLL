from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_dff(X, interval=50):
    """
    X : array (T, H, W)
    Affiche animation avec ΔF/F
    """

    # --- ΔF/F ---
    F0 = np.nanmedian(X, axis=0, keepdims=True)
    F0[F0 == 0] = 1  # éviter division par zéro

    X_dff = (X - F0) / F0

    # --- Setup animation ---
    T, H, W = X.shape
    fig, ax = plt.subplots()

    # Colormap adaptée (vert calcium)
    cmap = plt.cm.Greens.copy()
    cmap.set_bad(color='black')

    im = ax.imshow(X_dff[0], cmap=cmap, animated=True)

    # 🔥 IMPORTANT : limites robustes (évite outliers)
    vmin = np.nanpercentile(X_dff, 1)
    vmax = np.nanpercentile(X_dff, 99)
    im.set_clim(vmin, vmax)

    title = ax.set_title("Frame 0 (ΔF/F)")

    def update(frame):
        im.set_array(X_dff[frame])
        return [im]

    ani = FuncAnimation(
        fig,
        update,
        frames=T,
        interval=interval,
        blit=True
    )

    plt.show()

from pathlib import Path
import tifffile as tiff
from maitrise_curbd.io import load_dataset
from maitrise_curbd.masks import reduce_atlas_to_6_regions, subdivide_mask_by_spatial_clustering, build_parent_regions_dict

#########################################################################################################
# SETUP
##########################################################################################################

n_cohorte = 9
month = 14
souris = 410
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
    mouse=souris,
)

print("GCaMP :", gcamp.shape, gcamp.dtype)
print("Atlas :", atlas.shape, atlas.dtype)
print("ROI mask :", roi_mask.shape, roi_mask.dtype)

    
animate_dff(gcamp)