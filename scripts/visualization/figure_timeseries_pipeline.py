"""Compare les timeseries brute, pretraitee et reconstruite par le RNN."""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


### Permet de lancer le script direct sans installer le package a chaque fois.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from maitrise_curbd.io import get_data_root, load_dataset
from maitrise_curbd.masks import (
    reduce_atlas_to_6_regions,
    remove_thin_label_artifacts,
    subdivide_mask_by_spatial_clustering,
)
from maitrise_curbd.timeseries import (
    extract_timeseries_du_tenseur,
    regress_out_global_signal,
    smooth_timeseries,
)


def load_pickle_compatible(path):
    ### Les pkl Narval peuvent pointer vers numpy._core; petit pont local pour vieux NumPy.
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as exc:
        if exc.name != "numpy._core":
            raise

    import numpy.core as np_core

    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_core.numeric)

    with path.open("rb") as f:
        return pickle.load(f)


def robust_zscore(ts):
    ### Meme echelle visuelle pour comparer les formes, pas juste les amplitudes.
    ts = np.asarray(ts, dtype=float)
    center = np.nanmedian(ts, axis=1, keepdims=True)
    scale = np.nanpercentile(ts, 75, axis=1, keepdims=True) - np.nanpercentile(
        ts,
        25,
        axis=1,
        keepdims=True,
    )
    scale = np.where(scale <= 1e-12, np.nanstd(ts, axis=1, keepdims=True), scale)
    scale = np.where(scale <= 1e-12, 1.0, scale)
    return (ts - center) / scale


def sample_rnn(J, ts_processed, params):
    ### Simulation deterministic avec le J final: pas de re-training, juste "qu'est-ce que le RNN dessine".
    J = np.asarray(J, dtype=float)
    ts_processed = np.asarray(ts_processed, dtype=float)

    scale = np.nanmax(ts_processed)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Maximum invalide pour normaliser les timeseries pretraitees.")

    adata = np.clip(ts_processed / scale, -0.999, 0.999)

    dt_data = float(params["dtData"])
    dt_factor = int(params["dtFactor"])
    tau_rnn = float(params["tauRNN"])
    dt_rnn = dt_data / dt_factor

    t_data = dt_data * np.arange(adata.shape[1])
    t_rnn = np.arange(0, t_data[-1] + dt_rnn, dt_rnn)
    sample_idx = np.array([np.abs(t_rnn - t).argmin() for t in t_data], dtype=int)

    h = adata[:, 0].copy()
    rnn = np.zeros((adata.shape[0], len(t_rnn)), dtype=np.float32)
    rnn[:, 0] = np.tanh(h)

    for tt in range(1, len(t_rnn)):
        ### Dynamique RNN finale: tanh(H), interactions J, integration Euler.
        rnn[:, tt] = np.tanh(h)
        current = J.dot(rnn[:, tt])
        h = h + dt_rnn * (-h + current) / tau_rnn

    return rnn[:, sample_idx], adata


def get_rnn_reconstruction(pkl_data, ts_processed):
    ### Nouvelles runs: on prend la vraie trajectoire RNN sauvee dans le pkl.
    if pkl_data.get("RNN_final") is not None and pkl_data.get("Adata") is not None:
        params = pkl_data["parameters"]
        rnn = np.asarray(pkl_data["RNN_final"], dtype=float)
        adata = np.asarray(pkl_data["Adata"], dtype=float)

        if pkl_data.get("tData") is not None and pkl_data.get("tRNN") is not None:
            t_data = np.asarray(pkl_data["tData"], dtype=float)
            t_rnn = np.asarray(pkl_data["tRNN"], dtype=float)
            sample_idx = np.array([np.abs(t_rnn - t).argmin() for t in t_data], dtype=int)
        else:
            dt_factor = int(params["dtFactor"])
            sample_idx = np.arange(0, rnn.shape[1], dt_factor)[: adata.shape[1]]

        return rnn[:, sample_idx], adata, "RNN vrai de la run"

    ### Anciennes runs: on a J_final, mais pas le film RNN exact. On redessine donc en autonome.
    rnn, adata = sample_rnn(pkl_data["J_final"], ts_processed, pkl_data["parameters"])
    return rnn, adata, "RNN autonome depuis J final"


def build_pipeline_timeseries(pkl_data, data_root):
    params = pkl_data["parameters"]
    cohort = int(params["cohort"])
    month = int(params["month"])
    mouse = int(params["mouse"])

    ### Meme preprocessing que la loop: atlas clean, 6 regions, subdivision spatiale.
    gcamp, atlas, roi_mask = load_dataset(
        cohort=cohort,
        month=month,
        mouse=mouse,
        data_root=data_root,
    )
    gcamp = np.asarray(gcamp)
    atlas = np.asarray(atlas)
    roi_mask = np.asarray(roi_mask)

    clean_atlas = remove_thin_label_artifacts(
        atlas,
        size=int(params.get("atlas_clean_size", 5)),
        min_fraction=float(params.get("atlas_clean_min_fraction", 0.25)),
    )
    atlas_6 = reduce_atlas_to_6_regions(clean_atlas, roi_mask)
    sub_mask, _ = subdivide_mask_by_spatial_clustering(
        atlas_6,
        target_size=int(params["n_pixels"]),
    )

    ### Trace extraite avant GSR/lissage; ici DFF OFF veut dire fluorescence telle que recue.
    ts_raw = extract_timeseries_du_tenseur(gcamp, sub_mask).astype(np.float32)
    ts_processed = ts_raw.copy()

    if bool(params["use_global_regression"]):
        ts_processed = regress_out_global_signal(ts_processed)

    ts_processed = smooth_timeseries(
        ts_processed,
        sigma=float(params["lissage_sigma"]),
    ).astype(np.float32)

    ts_rnn, ts_target_scaled, rnn_label = get_rnn_reconstruction(pkl_data, ts_processed)

    return {
        "params": params,
        "ts_raw": ts_raw,
        "ts_processed": ts_processed,
        "ts_target_scaled": ts_target_scaled,
        "ts_rnn": ts_rnn,
        "rnn_label": rnn_label,
    }


def choose_regions(ts_processed, n_regions, mode):
    ### Par defaut on prend les traces qui bougent le plus: plus informatif que 8 traces plates.
    if mode == "first":
        return np.arange(min(n_regions, ts_processed.shape[0]))

    scores = np.nanstd(ts_processed, axis=1)
    return np.argsort(scores)[-n_regions:][::-1]


def save_figure(fig, output_dir, name):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=220)
    fig.savefig(output_dir / f"{name}.pdf")
    plt.close(fig)


def plot_pipeline(bundle, output_dir, n_regions, start, duration, region_mode):
    params = bundle["params"]
    ts_raw = bundle["ts_raw"]
    ts_processed = bundle["ts_processed"]
    ts_target_scaled = bundle["ts_target_scaled"]
    ts_rnn = bundle["ts_rnn"]
    rnn_label = bundle["rnn_label"]

    dt = float(params["dtData"])
    total_t = ts_raw.shape[1]
    start_idx = int(start / dt) if start is not None else 0
    duration_idx = int(duration / dt) if duration is not None else min(total_t, 25 * int(1 / dt))
    end_idx = min(total_t, start_idx + duration_idx)
    window = slice(start_idx, end_idx)
    t = np.arange(start_idx, end_idx) * dt

    regions = choose_regions(ts_processed, n_regions=n_regions, mode=region_mode)

    raw_z = robust_zscore(ts_raw[regions, window])
    proc_z = robust_zscore(ts_processed[regions, window])
    target_z = robust_zscore(ts_target_scaled[regions, window])
    rnn_z = robust_zscore(ts_rnn[regions, window])

    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(3, 3, height_ratios=[1.3, 1.1, 1.1])

    ### Panneau 1: quelques traces, meme region, trois versions du dessin.
    ax = fig.add_subplot(grid[0, :])
    offsets = np.arange(len(regions))[::-1] * 5.0
    for i, region in enumerate(regions):
        ax.plot(t, raw_z[i] + offsets[i], color="#8a8a8a", linewidth=1.0, alpha=0.7)
        ax.plot(t, proc_z[i] + offsets[i], color="#2f6f73", linewidth=1.4)
        ax.plot(t, rnn_z[i] + offsets[i], color="#c65f3a", linewidth=1.2)
        ax.text(t[0], offsets[i] + 1.8, f"region {region}", fontsize=8, color="0.25")

    ax.set_title("Memes regions: avant lissage, apres lissage et reconstruction RNN")
    ax.set_xlabel("Temps (s)")
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.18)
    ax.plot([], [], color="#8a8a8a", label="avant lissage")
    ax.plot([], [], color="#2f6f73", label="apres lissage")
    ax.plot([], [], color="#c65f3a", label=rnn_label)
    ax.legend(frameon=False, ncol=3, loc="upper right")

    ### Panneau 2: heatmaps, utile pour voir le dessin global sur toutes les regions choisies.
    panels = [
        ("Avant lissage", raw_z),
        ("Apres lissage", proc_z),
        (rnn_label, rnn_z),
    ]
    vmax = np.nanpercentile(np.abs(np.vstack([raw_z, proc_z, rnn_z])), 98)
    vmax = max(vmax, 1.0)
    for j, (title, arr) in enumerate(panels):
        ax_h = fig.add_subplot(grid[1, j])
        im = ax_h.imshow(
            arr,
            aspect="auto",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax_h.set_title(title)
        ax_h.set_xlabel("Temps dans la fenetre")
        if j == 0:
            ax_h.set_ylabel("Regions choisies")
        else:
            ax_h.set_yticks([])
    fig.colorbar(im, ax=[fig.axes[-3], fig.axes[-2], fig.axes[-1]], shrink=0.75, label="z robuste")

    ### Panneau 3: erreur RNN vs signal cible scale. Ca dit ou le modele manque le dessin.
    err = rnn_z - target_z
    ax_e = fig.add_subplot(grid[2, 0])
    im_e = ax_e.imshow(
        err,
        aspect="auto",
        cmap="PiYG",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax_e.set_title("Erreur RNN - cible")
    ax_e.set_xlabel("Temps dans la fenetre")
    ax_e.set_ylabel("Regions choisies")
    fig.colorbar(im_e, ax=ax_e, shrink=0.75, label="erreur z")

    ax_m = fig.add_subplot(grid[2, 1:])
    rmse_by_t = np.sqrt(np.nanmean(err**2, axis=0))
    ax_m.plot(t, rmse_by_t, color="#4b4b4b", linewidth=1.5)
    ax_m.set_title("Erreur moyenne dans le temps")
    ax_m.set_xlabel("Temps (s)")
    ax_m.set_ylabel("RMSE robuste")
    ax_m.grid(True, alpha=0.25)

    gsr = "ON" if bool(params["use_global_regression"]) else "OFF"
    fig.suptitle(
        "Pipeline visuel | "
        f"C{params['cohort']} M{params['month']} souris {params['mouse']} | "
        f"pix={params['n_pixels']} sigma={params['lissage_sigma']} GSR {gsr}",
        fontsize=14,
    )
    save_figure(fig, output_dir, "timeseries_pipeline")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-regions", type=int, default=8)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--region-mode", choices=["variance", "first"], default="variance")
    return parser.parse_args()


def main():
    args = parse_args()
    pkl_data = load_pickle_compatible(args.pkl)
    data_root = get_data_root(args.data_root)
    output_dir = args.output_dir or args.pkl.parent / "timeseries_pipeline_figures" / args.pkl.stem

    bundle = build_pipeline_timeseries(pkl_data, data_root)
    plot_pipeline(
        bundle,
        output_dir=output_dir,
        n_regions=args.n_regions,
        start=args.start_sec,
        duration=args.duration_sec,
        region_mode=args.region_mode,
    )

    print(f"Figure sauvegardee dans: {output_dir}")


if __name__ == "__main__":
    main()
