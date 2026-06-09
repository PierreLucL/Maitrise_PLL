import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.linalg import svd
from sklearn.decomposition import PCA
import curbd

real_ts = np.load("real_ts.npy")


sim = curbd.threeRegionSim(number_units=35,dtData=0.01,T=12.4)
activity = np.concatenate((sim['Ra'], sim['Rb'], sim['Rc']), 0)
toy_ts = activity[:, :-1]
toy_ts = (toy_ts - toy_ts.mean(axis=1, keepdims=True)) / toy_ts.std(axis=1, keepdims=True)
real_ts = (real_ts - real_ts.mean(axis=1, keepdims=True)) / real_ts.std(axis=1, keepdims=True)

# ============================================================
# INPUTS
# ============================================================
# toy_ts  : numpy array (N x T)
# real_ts : numpy array (N x T)

# Example:
# toy_ts = toy_model_rates
# real_ts = experimental_rates

# ============================================================
# PARAMETERS
# ============================================================
fs = 1.0  # sampling frequency (change if needed)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_psd_mean(ts, fs=1.0):
    psds = []

    for i in range(ts.shape[0]):
        f, Pxx = welch(ts[i], fs=fs, nperseg=min(256, ts.shape[1]))
        psds.append(Pxx)

    psds = np.array(psds)
    return f, np.mean(psds, axis=0)


def compute_mean_autocorr(ts, max_lag=200):

    acs = []

    for i in range(ts.shape[0]):

        x = ts[i]
        x = x - np.mean(x)

        ac = np.correlate(x, x, mode='full')
        ac = ac[len(ac)//2:]

        ac /= ac[0]

        acs.append(ac[:max_lag])

    acs = np.array(acs)

    return np.mean(acs, axis=0)


def compute_pca(ts):

    pca = PCA()
    pca.fit(ts.T)

    cumvar = np.cumsum(pca.explained_variance_ratio_)

    return cumvar


def compute_svd_spectrum(ts):

    U, S, Vh = svd(ts, full_matrices=False)

    S = S / np.sum(S)

    return S


def compute_corr(ts):

    return np.corrcoef(ts)


def compute_derivatives(ts):

    dts = np.diff(ts, axis=1)

    return dts.flatten()

print('ici')
dx_toy = np.diff(toy_ts, axis=1)
dx_real = np.diff(real_ts, axis=1)

print(np.std(dx_toy))
print(np.std(dx_real))
print(np.std(dx_toy)/np.std(dx_real))
print("std(x) toy :", np.std(toy_ts))
print("std(x) real:", np.std(real_ts))

print("relative dx toy :",
      np.std(dx_toy) / np.std(toy_ts))

print("relative dx real:",
      np.std(dx_real) / np.std(real_ts))

print("ratio relative:",
      (np.std(dx_toy) / np.std(toy_ts)) /
      (np.std(dx_real) / np.std(real_ts)))
# ============================================================
# COMPUTATIONS
# ============================================================

datasets = {
    "Toy model": toy_ts,
    "Real data": real_ts
}

results = {}

for name, ts in datasets.items():

    ts = np.nan_to_num(ts)

    results[name] = {}

    # PCA
    results[name]["pca"] = compute_pca(ts)

    # Correlation matrix
    results[name]["corr"] = compute_corr(ts)

    # PSD
    results[name]["freq"], results[name]["psd"] = compute_psd_mean(ts, fs)

    # Autocorrelation
    results[name]["ac"] = compute_mean_autocorr(ts)

    # Derivatives
    results[name]["deriv"] = compute_derivatives(ts)

    # SVD
    results[name]["svd"] = compute_svd_spectrum(ts)


# ============================================================
# PLOTTING
# ============================================================

fig, axes = plt.subplots(6, 2, figsize=(12, 22))

for col, name in enumerate(datasets.keys()):

    # --------------------------------------------------------
    # PCA cumulative variance
    # --------------------------------------------------------
    axes[0, col].plot(results[name]["pca"])
    axes[0, col].set_title(f"{name} - PCA cumulative variance")
    axes[0, col].set_xlabel("Number of PCs")
    axes[0, col].set_ylabel("Cumulative variance")

    # --------------------------------------------------------
    # Correlation matrix
    # --------------------------------------------------------
    im = axes[1, col].imshow(
        results[name]["corr"],
        vmin=-1,
        vmax=1,
        cmap='coolwarm'
    )

    axes[1, col].set_title(f"{name} - Correlation matrix")

    # --------------------------------------------------------
    # PSD
    # --------------------------------------------------------
    axes[2, col].loglog(
        results[name]["freq"],
        results[name]["psd"]
    )

    axes[2, col].set_title(f"{name} - Mean PSD")
    axes[2, col].set_xlabel("Frequency")
    axes[2, col].set_ylabel("Power")

    # --------------------------------------------------------
    # Autocorrelation
    # --------------------------------------------------------
    axes[3, col].plot(results[name]["ac"])

    axes[3, col].set_title(f"{name} - Mean autocorrelation")
    axes[3, col].set_xlabel("Lag")
    axes[3, col].set_ylabel("Correlation")

    # --------------------------------------------------------
    # Histogram of derivatives
    # --------------------------------------------------------
    axes[4, col].hist(
        results[name]["deriv"],
        bins=100,
        density=True
    )

    axes[4, col].set_title(f"{name} - Derivative histogram")
    axes[4, col].set_xlabel("dx/dt")

    # --------------------------------------------------------
    # SVD spectrum
    # --------------------------------------------------------
    axes[5, col].semilogy(results[name]["svd"])

    axes[5, col].set_title(f"{name} - Singular values")
    axes[5, col].set_xlabel("Component rank")
    axes[5, col].set_ylabel("Normalized singular value")


plt.tight_layout()
plt.show()