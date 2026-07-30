from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_dff(X, roi_mask, interval=50):
    X = np.asarray(X, dtype=np.float32)
    roi_mask = np.asarray(roi_mask, dtype=bool)

    if roi_mask.shape != X.shape[1:]:
        raise ValueError(
            f"roi_mask {roi_mask.shape} incompatible avec X {X.shape}"
        )

    X = np.where(roi_mask[None, :, :], X, np.nan)

    F0 = np.nanmedian(X, axis=0, keepdims=True)
    F0[(F0 == 0) | ~np.isfinite(F0)] = np.nan

    X_dff = (X - F0) / F0

    T = X_dff.shape[0]

    cmap = plt.cm.Greens.copy()
    cmap.set_bad("black")

    vmin = np.nanpercentile(X_dff, 1)
    vmax = np.nanpercentile(X_dff, 99)

    fig, ax = plt.subplots()

    im = ax.imshow(
        X_dff[0],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        animated=True,
    )

    title = ax.set_title("Frame 0 — ΔF/F")
    ax.axis("off")

    def update(frame):
        im.set_data(X_dff[frame])
        title.set_text(f"Frame {frame}/{T - 1} — ΔF/F")
        return im, title

    ani = FuncAnimation(
        fig,
        update,
        frames=T,
        interval=interval,
        blit=True,
    )

    plt.show()
    return ani

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

ani = animate_dff(gcamp, roi_mask)