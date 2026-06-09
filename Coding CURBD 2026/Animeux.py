from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Pipeline import load_gcamp

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

### Liste des souris disponibles ###
souris = ['M387-6', 'M396-6', 'M410-6', 'M412-8']


### Choix de la souris ###
Idx_souris = 0
    
dataset,mask = load_gcamp(f"/Users/pierre-luclarouche/Desktop/École/Maîtrise/CURBD-master/Data H5/{souris[Idx_souris]}_v4_mvmt.h5")
animate_dff(dataset)