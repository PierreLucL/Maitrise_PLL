"""Evaluation severe des reconstructions RNN sauvegardees dans les pkl."""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


### On garde les couleurs constantes pour que l'oeil apprenne vite les figures.
TARGET_COLOR = "#2f6f73"
RNN_COLOR = "#c65f3a"
BAD_COLOR = "#9b3f3f"
GOOD_COLOR = "#3f7f5f"


def load_pickle_compatible(path):
    ### Les pkl Narval peuvent venir d'un numpy plus recent. Petit adaptateur maison.
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


def sample_rnn_on_data_grid(data):
    ### RNN_final vit sur tRNN; Adata vit sur tData. On remet tout sur la grille data.
    rnn = np.asarray(data["RNN_final"], dtype=float)
    target = np.asarray(data["Adata"], dtype=float)

    if data.get("tData") is not None and data.get("tRNN") is not None:
        t_data = np.asarray(data["tData"], dtype=float)
        t_rnn = np.asarray(data["tRNN"], dtype=float)
        sample_idx = np.array([np.abs(t_rnn - t).argmin() for t in t_data], dtype=int)
    else:
        dt_factor = int(data["parameters"]["dtFactor"])
        sample_idx = np.arange(0, rnn.shape[1], dt_factor)[: target.shape[1]]

    pred = rnn[:, sample_idx]
    n_time = min(target.shape[1], pred.shape[1])
    return target[:, :n_time], pred[:, :n_time]


def corr_1d(a, b):
    ### Correlation robuste aux traces plates. Trace plate = pas de signal a evaluer, donc NaN.
    ok = np.isfinite(a) & np.isfinite(b)
    if np.sum(ok) < 3:
        return np.nan
    a = a[ok]
    b = b[ok]
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    return float(np.sum(a * b) / denom) if denom > 1e-12 else np.nan


def robust_zscore(ts):
    ### Z-score par IQR: moins impressionnable quand une trace a un spike trop intense.
    center = np.nanmedian(ts, axis=1, keepdims=True)
    q75 = np.nanpercentile(ts, 75, axis=1, keepdims=True)
    q25 = np.nanpercentile(ts, 25, axis=1, keepdims=True)
    scale = q75 - q25
    fallback = np.nanstd(ts, axis=1, keepdims=True)
    scale = np.where(scale <= 1e-12, fallback, scale)
    scale = np.where(scale <= 1e-12, 1.0, scale)
    return (ts - center) / scale


def event_hit_rate(target, pred, percentile=95, tolerance=2):
    ### Check "pics": est-ce que le RNN allume proche des gros evenements de la cible?
    target_abs = np.abs(target)
    pred_abs = np.abs(pred)
    threshold_target = np.nanpercentile(target_abs, percentile)
    threshold_pred = np.nanpercentile(pred_abs, percentile)

    target_events = np.flatnonzero(target_abs >= threshold_target)
    pred_events = pred_abs >= threshold_pred

    if target_events.size == 0 or not np.any(pred_events):
        return np.nan

    expanded = pred_events.copy()
    for shift in range(1, tolerance + 1):
        expanded[shift:] |= pred_events[:-shift]
        expanded[:-shift] |= pred_events[shift:]

    return float(np.mean(expanded[target_events]))


def per_region_metrics(target, pred, params, pkl_name):
    ### Toutes les metriques qui punissent le "beau score mais dessin mou".
    rows = []
    dz_target = np.diff(target, axis=1)
    dz_pred = np.diff(pred, axis=1)

    for region in range(target.shape[0]):
        y = target[region]
        yhat = pred[region]
        denom = np.nansum((y - np.nanmean(y)) ** 2)
        sse = np.nansum((y - yhat) ** 2)
        pvar_region = 1 - sse / denom if denom > 1e-12 else np.nan

        rmse = float(np.sqrt(np.nanmean((y - yhat) ** 2)))
        corr = corr_1d(y, yhat)
        deriv_corr = corr_1d(dz_target[region], dz_pred[region])
        peak_hit = event_hit_rate(y, yhat)

        rows.append(
            {
                "pkl": pkl_name,
                "region": region,
                "cohort": int(params["cohort"]),
                "month": int(params["month"]),
                "mouse": int(params["mouse"]),
                "n_pixels": int(params["n_pixels"]),
                "lissage_sigma": float(params["lissage_sigma"]),
                "g": float(params["g"]),
                "tauRNN": float(params["tauRNN"]),
                "dtFactor": int(params["dtFactor"]),
                "use_global_regression": bool(params["use_global_regression"]),
                "corr": corr,
                "deriv_corr": deriv_corr,
                "pVar_region": float(pvar_region),
                "rmse": rmse,
                "peak_hit_rate": peak_hit,
                "target_std": float(np.nanstd(y)),
                "pred_std": float(np.nanstd(yhat)),
            }
        )

    return pd.DataFrame(rows)


def summarize_metrics(metrics, row):
    ### Resume par pkl: medianes et bas de distribution, plus utiles qu'un seul score global.
    summary = {
        "pkl": metrics["pkl"].iloc[0],
        "cohort": metrics["cohort"].iloc[0],
        "month": metrics["month"].iloc[0],
        "mouse": metrics["mouse"].iloc[0],
        "n_pixels": metrics["n_pixels"].iloc[0],
        "lissage_sigma": metrics["lissage_sigma"].iloc[0],
        "g": metrics["g"].iloc[0],
        "tauRNN": metrics["tauRNN"].iloc[0],
        "dtFactor": metrics["dtFactor"].iloc[0],
        "use_global_regression": metrics["use_global_regression"].iloc[0],
        "pVar_finale": np.nan,
        "pVar_train_end": np.nan,
        "pVar_free_mean": np.nan,
    }

    if isinstance(row, dict):
        for key in ["pVar_finale", "pVar_train_end", "pVar_free_mean"]:
            summary[key] = row.get(key, np.nan)

    for metric in ["corr", "deriv_corr", "pVar_region", "rmse", "peak_hit_rate"]:
        values = metrics[metric].to_numpy(dtype=float)
        summary[f"{metric}_mean"] = float(np.nanmean(values))
        summary[f"{metric}_median"] = float(np.nanmedian(values))
        summary[f"{metric}_p10"] = float(np.nanpercentile(values, 10))
        summary[f"{metric}_p90"] = float(np.nanpercentile(values, 90))

    return summary


def save_fig(fig, output_dir, name):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=220)
    fig.savefig(output_dir / f"{name}.pdf")
    plt.close(fig)


def label_from_summary(row):
    gsr = "GSR ON" if row["use_global_regression"] else "GSR OFF"
    return (
        f"M{int(row['mouse'])} | g={row['g']:.1f} | "
        f"sigma={row['lissage_sigma']:.0f} | {gsr}"
    )


def plot_summary(summary, output_dir):
    ### Vue executive: pVar vs metriques qui regardent vraiment le dessin.
    summary = summary.sort_values(["mouse", "g", "lissage_sigma"]).copy()
    labels = [label_from_summary(row) for _, row in summary.iterrows()]
    x = np.arange(len(summary))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True)
    specs = [
        ("pVar_finale", "pVar global"),
        ("corr_median", "Correlation mediane par region"),
        ("deriv_corr_median", "Correlation mediane des derivees"),
        ("peak_hit_rate_median", "Hit-rate median des gros evenements"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), specs):
        values = summary[metric].to_numpy(dtype=float)
        colors = np.where(values >= np.nanmedian(values), GOOD_COLOR, BAD_COLOR)
        ax.bar(x, values, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        for xi, yi in zip(x, values):
            if np.isfinite(yi):
                ax.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)

    axes[-1, 0].set_xticks(x, labels, rotation=45, ha="right")
    axes[-1, 1].set_xticks(x, labels, rotation=45, ha="right")
    fig.suptitle("Reconstruction RNN: score global vs qualite du dessin", fontsize=14)
    save_fig(fig, output_dir, "summary_reconstruction_metrics")


def plot_metric_distributions(metrics, output_dir):
    ### Distribution par region: si la mediane est belle mais que le bas est laid, on le voit.
    metrics = metrics.copy()
    metrics["label"] = [
        label_from_summary(row)
        for _, row in metrics[
            ["mouse", "g", "lissage_sigma", "use_global_regression"]
        ].iterrows()
    ]
    labels = list(dict.fromkeys(metrics["label"]))

    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), sharex=True)
    specs = [
        ("corr", "Correlation par region"),
        ("deriv_corr", "Correlation des derivees par region"),
        ("pVar_region", "pVar par region"),
        ("peak_hit_rate", "Hit-rate des gros evenements"),
    ]

    for ax, (metric, title) in zip(axes.ravel(), specs):
        data = [metrics.loc[metrics["label"] == label, metric].dropna() for label in labels]
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Distribution des scores region par region", fontsize=14)
    save_fig(fig, output_dir, "distribution_reconstruction_metrics")


def plot_worst_regions(target, pred, metrics, output_dir, pkl_stem, n_regions=6, duration_sec=30):
    ### On montre volontairement les pires regions: c'est la figure pas complaisante.
    params = {
        "mouse": int(metrics["mouse"].iloc[0]),
        "g": float(metrics["g"].iloc[0]),
        "sigma": float(metrics["lissage_sigma"].iloc[0]),
        "gsr": bool(metrics["use_global_regression"].iloc[0]),
        "dtData": 1 / 12,
    }

    ranked = metrics.sort_values(["deriv_corr", "corr"], ascending=True)
    regions = ranked["region"].head(n_regions).to_numpy(dtype=int)
    n_time = min(target.shape[1], int(duration_sec / params["dtData"]))
    window = slice(0, n_time)
    t = np.arange(n_time) * params["dtData"]

    target_z = robust_zscore(target[regions, window])
    pred_z = robust_zscore(pred[regions, window])
    d_target = robust_zscore(np.diff(target[regions, window], axis=1))
    d_pred = robust_zscore(np.diff(pred[regions, window], axis=1))
    t_diff = t[1:]

    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.25, 1.0])

    ax_signal = fig.add_subplot(grid[0, :])
    offsets = np.arange(len(regions))[::-1] * 5.0
    for i, region in enumerate(regions):
        ax_signal.plot(t, target_z[i] + offsets[i], color=TARGET_COLOR, linewidth=1.4)
        ax_signal.plot(t, pred_z[i] + offsets[i], color=RNN_COLOR, linewidth=1.1)
        ax_signal.text(t[0], offsets[i] + 1.7, f"region {region}", fontsize=8)
    ax_signal.plot([], [], color=TARGET_COLOR, label="cible traitee")
    ax_signal.plot([], [], color=RNN_COLOR, label="RNN vrai")
    ax_signal.set_title("Pires regions selon derivee/correlation: signal")
    ax_signal.set_yticks([])
    ax_signal.set_xlabel("Temps (s)")
    ax_signal.grid(True, axis="x", alpha=0.18)
    ax_signal.legend(frameon=False, ncol=2, loc="upper right")

    ax_deriv = fig.add_subplot(grid[1, :])
    for i, region in enumerate(regions):
        ax_deriv.plot(t_diff, d_target[i] + offsets[i], color=TARGET_COLOR, linewidth=1.2)
        ax_deriv.plot(t_diff, d_pred[i] + offsets[i], color=RNN_COLOR, linewidth=1.0)
    ax_deriv.set_title("Memes regions: derivee temporelle")
    ax_deriv.set_yticks([])
    ax_deriv.set_xlabel("Temps (s)")
    ax_deriv.grid(True, axis="x", alpha=0.18)

    ax_scatter = fig.add_subplot(grid[2, 0])
    ax_scatter.scatter(metrics["corr"], metrics["deriv_corr"], s=12, alpha=0.55, color="#444444")
    ax_scatter.scatter(
        metrics.loc[metrics["region"].isin(regions), "corr"],
        metrics.loc[metrics["region"].isin(regions), "deriv_corr"],
        s=28,
        color=BAD_COLOR,
        label="regions montrees",
    )
    ax_scatter.axhline(0, color="black", linewidth=0.8)
    ax_scatter.axvline(0, color="black", linewidth=0.8)
    ax_scatter.set_xlabel("Correlation signal")
    ax_scatter.set_ylabel("Correlation derivee")
    ax_scatter.set_title("Chaque point = une region")
    ax_scatter.grid(True, alpha=0.25)
    ax_scatter.legend(frameon=False)

    ax_hist = fig.add_subplot(grid[2, 1])
    ax_hist.hist(metrics["deriv_corr"].dropna(), bins=35, color="#666666", alpha=0.85)
    ax_hist.axvline(metrics["deriv_corr"].median(), color=RNN_COLOR, linewidth=2, label="mediane")
    ax_hist.set_xlabel("Correlation des derivees")
    ax_hist.set_ylabel("Nombre de regions")
    ax_hist.set_title("Est-ce que le RNN suit les mouvements rapides?")
    ax_hist.legend(frameon=False)

    gsr = "ON" if params["gsr"] else "OFF"
    fig.suptitle(
        f"M{params['mouse']} | g={params['g']:.1f} | "
        f"sigma={params['sigma']:.0f} | {gsr}",
        fontsize=14,
    )
    save_fig(fig, output_dir / "worst_regions", f"{pkl_stem}_worst_regions")


def find_pkls(paths):
    ### Accepte un dossier de run ou une liste de pkl, parce qu'on veut pas gosser.
    pkls = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            pkls.extend(sorted(path.glob("config*.pkl")))
        elif path.suffix == ".pkl":
            pkls.append(path)
    return sorted(dict.fromkeys(pkls))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--worst-regions", type=int, default=6)
    parser.add_argument("--duration-sec", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    pkls = find_pkls(args.paths)
    if not pkls:
        raise SystemExit("Aucun pkl trouve.")

    output_dir = args.output_dir or pkls[0].parent / "rnn_reconstruction_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    summaries = []

    for pkl_path in pkls:
        data = load_pickle_compatible(pkl_path)
        if data.get("RNN_final") is None or data.get("Adata") is None:
            print(f"Skip sans RNN_final/Adata: {pkl_path}")
            continue

        target, pred = sample_rnn_on_data_grid(data)
        metrics = per_region_metrics(
            target=target,
            pred=pred,
            params=data["parameters"],
            pkl_name=pkl_path.name,
        )
        all_metrics.append(metrics)
        summaries.append(summarize_metrics(metrics, data.get("row", {})))

        plot_worst_regions(
            target=target,
            pred=pred,
            metrics=metrics,
            output_dir=output_dir,
            pkl_stem=pkl_path.stem,
            n_regions=args.worst_regions,
            duration_sec=args.duration_sec,
        )

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    metrics_df.to_csv(output_dir / "per_region_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)

    plot_summary(summary_df, output_dir)
    plot_metric_distributions(metrics_df, output_dir)

    print(f"Evaluation sauvegardee dans: {output_dir}")
    print(summary_df.sort_values("pVar_finale", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
