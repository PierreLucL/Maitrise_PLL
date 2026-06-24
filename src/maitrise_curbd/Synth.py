import numpy as np
from scipy.ndimage import gaussian_filter1d

### Synthétiques ?

def generate_synthetic_gcamp_rnn(
    height=40,
    width=60,
    T=1000,
    g=1.5,
    dt=0.33,
    tau=1.0,
    noise_std=0.02,
    same_region_strength=2.5,
    spatial_decay=8.0,
    calcium_tau=1.2,
    apply_gcamp=True,
    seed=None,
):
    """
    Génère des données synthétiques type GCaMP avec 6 régions carrées.

    Returns
    -------
    mask : array (height, width)
        Masque contenant les labels 0 à 5.
    W : array (N, N)
        Matrice de connectivité vraie entre tous les pixels.
    X : array (N, T)
        Séries temporelles simulées, N pixels x T temps.
    """

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------------
    # 1. Masque : 6 régions en grille 2 x 3
    # ---------------------------------------------------------------------
    mask = np.zeros((height, width), dtype=int)

    n_rows = 2
    n_cols = 3

    region_h = height // n_rows
    region_w = width // n_cols

    label = 0
    for i in range(n_rows):
        for j in range(n_cols):
            y0 = i * region_h
            y1 = (i + 1) * region_h if i < n_rows - 1 else height
            x0 = j * region_w
            x1 = (j + 1) * region_w if j < n_cols - 1 else width

            mask[y0:y1, x0:x1] = label
            label += 1

    N = height * width
    labels = mask.ravel()

    # Coordonnées spatiales des pixels
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    coords = np.column_stack([yy.ravel(), xx.ravel()])

    # ---------------------------------------------------------------------
    # 2. Matrice de connectivité structurée
    # ---------------------------------------------------------------------
    dy = coords[:, 0][:, None] - coords[:, 0][None, :]
    dx = coords[:, 1][:, None] - coords[:, 1][None, :]
    dist = np.sqrt(dx**2 + dy**2)

    spatial_kernel = np.exp(-dist / spatial_decay)

    same_region = labels[:, None] == labels[None, :]

    structure = spatial_kernel.copy()
    structure[same_region] *= same_region_strength

    # Poids aléatoires structurés
    W = rng.normal(0, 1, size=(N, N)).astype(np.float32)
    W *= structure.astype(np.float32)

    # Pas d'auto-connexion directe
    np.fill_diagonal(W, 0)

    # Normalisation par rayon spectral
    eigvals = np.linalg.eigvals(W)
    spectral_radius = np.max(np.abs(eigvals))

    if spectral_radius > 0:
        W = W / spectral_radius * g

    W = W.astype(np.float32)

    # ---------------------------------------------------------------------
    # 3. Simulation RNN
    # ---------------------------------------------------------------------
    H = rng.normal(0, 0.1, size=N).astype(np.float32)
    X_neural = np.zeros((N, T), dtype=np.float32)

    for t in range(T):
        R = np.tanh(H)
        X_neural[:, t] = R

        noise = rng.normal(0, noise_std, size=N).astype(np.float32)

        dH = (-H + W @ R + noise) / tau
        H = H + dt * dH

    # ---------------------------------------------------------------------
    # 4. Observation GCaMP synthétique
    # ---------------------------------------------------------------------
    if apply_gcamp:
        # filtre exponentiel approximé par un lissage gaussien temporel
        sigma_frames = calcium_tau / dt
        X = gaussian_filter1d(X_neural, sigma=sigma_frames, axis=1)

        # bruit observationnel calcium
        X += rng.normal(0, noise_std, size=X.shape).astype(np.float32)

        # normalisation par pixel
        X = X - X.mean(axis=1, keepdims=True)
        X = X / (X.std(axis=1, keepdims=True) + 1e-8)

    else:
        X = X_neural

    return mask, W, X