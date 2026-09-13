"""Figures pour comprendre le screening preprocessing M6 lance sur Narval."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


### Style pas trop fancy: on veut comprendre vite sans se faire aveugler par du flafla.
PALETTE_GSR = {False: "#2f6f73", True: "#c65f3a"}


def read_summary(csv_path):
    ### On charge juste les runs terminees, parce que comparer avec des erreurs c'est du sport inutile.
    data = pd.read_csv(csv_path)
    data = data[data["status"] == "done"].copy()
    data["gsr_label"] = np.where(data["use_global_regression"], "GSR ON", "GSR OFF")
    data["combo"] = (
        "pix "
        + data["n_pixels"].astype(str)
        + " | sigma "
        + data["lissage_sigma"].astype(str)
        + " | "
        + data["gsr_label"]
    )
    return data


def save_figure(fig, output_dir, name):
    ### PNG pour VSCode, PDF pour quand on veut une version propre dans un rapport.
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=220)
    fig.savefig(output_dir / f"{name}.pdf")
    plt.close(fig)


def plot_best_per_mouse(data, output_dir):
    ### Ici on garde seulement le meilleur setting par souris: le leaderboard simple.
    best = (
        data.sort_values("pVar_finale", ascending=False)
        .groupby("mouse", as_index=False)
        .head(1)
        .sort_values("pVar_finale", ascending=True)
    )
    best.to_csv(output_dir / "best_per_mouse.csv", index=False)

    colors = [PALETTE_GSR[v] for v in best["use_global_regression"]]
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.barh(best["mouse"].astype(str), best["pVar_finale"], color=colors)
    ax.axvline(0.7, color="0.35", linestyle="--", linewidth=1, label="pVar = 0.70")
    for y, row in enumerate(best.itertuples(index=False)):
        ax.text(
            row.pVar_finale + 0.008,
            y,
            f"{row.pVar_finale:.3f}  pix={row.n_pixels}, s={row.lissage_sigma}, {row.gsr_label}",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, min(0.9, max(0.86, best["pVar_finale"].max() + 0.08)))
    ax.set_xlabel("pVar finale")
    ax.set_ylabel("Souris")
    ax.set_title("Meilleur pVar par souris")
    ax.legend(loc="lower right", frameon=False)
    save_figure(fig, output_dir, "01_best_pvar_per_mouse")


def plot_combo_summary(data, output_dir):
    ### Mediane > moyenne ici, parce que deux-trois runs partent dans le champ solide.
    summary = (
        data.groupby(["n_pixels", "lissage_sigma", "use_global_regression"], as_index=False)
        .agg(
            pVar_median=("pVar_finale", "median"),
            pVar_mean=("pVar_finale", "mean"),
            pVar_min=("pVar_finale", "min"),
            pVar_max=("pVar_finale", "max"),
        )
        .sort_values("pVar_median", ascending=False)
    )
    summary["gsr_label"] = np.where(summary["use_global_regression"], "GSR ON", "GSR OFF")
    summary["label"] = (
        "pix="
        + summary["n_pixels"].astype(str)
        + ", s="
        + summary["lissage_sigma"].astype(str)
        + ", "
        + summary["gsr_label"]
    )
    summary.to_csv(output_dir / "combo_summary.csv", index=False)

    top = summary.head(12).sort_values("pVar_median", ascending=True)
    colors = [PALETTE_GSR[v] for v in top["use_global_regression"]]
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.barh(top["label"], top["pVar_median"], color=colors)
    for y, row in enumerate(top.itertuples(index=False)):
        ax.text(row.pVar_median + 0.008, y, f"med={row.pVar_median:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, min(0.9, max(0.82, top["pVar_median"].max() + 0.09)))
    ax.set_xlabel("pVar mediane sur les 13 souris")
    ax.set_ylabel("Combo preprocessing")
    ax.set_title("Top combos preprocessing, robuste aux outliers")
    save_figure(fig, output_dir, "02_top_combos_by_median")


def plot_pixels_sigma_heatmap(data, output_dir):
    ### Heatmap compacte: on voit direct que 50 pixels + sigma 4 gagne le tournoi.
    pivot = data.pivot_table(
        index="n_pixels",
        columns="lissage_sigma",
        values="pVar_finale",
        aggfunc="median",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=max(0, np.nanmin(pivot.values)),
        vmax=np.nanmax(pivot.values),
        cbar_kws={"label": "pVar mediane"},
        ax=ax,
    )
    ax.set_xlabel("Sigma du lissage")
    ax.set_ylabel("Pixels par sous-region")
    ax.set_title("Effet n_pixels x sigma")
    save_figure(fig, output_dir, "03_heatmap_pixels_sigma")


def plot_gsr_effect_at_winner(data, output_dir):
    ### Focus sur le combo gagnant: n_pixels=50 et sigma=4, puis GSR ON vs OFF souris par souris.
    winner = data[(data["n_pixels"] == 50) & (data["lissage_sigma"] == 4)].copy()
    pivot = winner.pivot_table(
        index="mouse",
        columns="gsr_label",
        values="pVar_finale",
        aggfunc="first",
    ).sort_index()
    pivot["delta_on_minus_off"] = pivot["GSR ON"] - pivot["GSR OFF"]
    pivot.to_csv(output_dir / "gsr_effect_at_npix50_sigma4.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1.35, 1]})

    plot_data = winner.sort_values(["mouse", "use_global_regression"])
    sns.lineplot(
        data=plot_data,
        x="gsr_label",
        y="pVar_finale",
        hue="mouse",
        marker="o",
        palette="tab20",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("pVar finale")
    axes[0].set_title("GSR ON/OFF au combo gagnant")
    axes[0].grid(True, axis="y", alpha=0.25)

    delta = pivot.sort_values("delta_on_minus_off")
    colors = np.where(delta["delta_on_minus_off"] >= 0, PALETTE_GSR[True], PALETTE_GSR[False])
    axes[1].barh(delta.index.astype(str), delta["delta_on_minus_off"], color=colors)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Delta pVar: GSR ON - GSR OFF")
    axes[1].set_ylabel("Souris")
    axes[1].set_title("Qui aime le GSR?")
    axes[1].grid(True, axis="x", alpha=0.25)

    save_figure(fig, output_dir, "04_gsr_effect_npix50_sigma4")


def plot_outliers(data, output_dir):
    ### Cette figure explique pourquoi la moyenne raconte n'importe quoi dans certains cas.
    clipped = data.copy()
    clipped["pVar_clipped"] = clipped["pVar_finale"].clip(lower=-0.25)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
    sns.boxplot(
        data=clipped,
        x="lissage_sigma",
        y="pVar_clipped",
        hue="gsr_label",
        palette={"GSR OFF": PALETTE_GSR[False], "GSR ON": PALETTE_GSR[True]},
        ax=axes[0],
    )
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xlabel("Sigma du lissage")
    axes[0].set_ylabel("pVar finale, clippee a -0.25")
    axes[0].set_title("Distribution des pVar par sigma")
    axes[0].legend(title="", frameon=False)

    worst = data.sort_values("pVar_finale").head(8).sort_values("pVar_finale", ascending=True)
    labels = [
        f"{int(r.mouse)} | pix={int(r.n_pixels)} s={int(r.lissage_sigma)} {r.gsr_label}"
        for r in worst.itertuples(index=False)
    ]
    axes[1].barh(labels, worst["pVar_finale"], color="#8b3a3a")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_xlabel("pVar finale")
    axes[1].set_title("Pires runs, pour spotter les explosions")
    save_figure(fig, output_dir, "05_outliers_and_sigma")


def plot_runtime_tradeoff(data, output_dir):
    ### Petit reality check: 50 pixels performe mieux, mais coute plus cher en temps.
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    sns.scatterplot(
        data=data,
        x="n_subregions",
        y="pVar_finale",
        hue="n_pixels",
        style="lissage_sigma",
        palette="viridis",
        s=72,
        ax=ax,
    )
    ax.axhline(0.7, color="0.35", linestyle="--", linewidth=1)
    ax.set_xlabel("Nombre de sous-regions")
    ax.set_ylabel("pVar finale")
    ax.set_title("Performance vs granularite spatiale")
    ax.legend(title="n_pixels / sigma", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, output_dir, "06_pvar_vs_subregions")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/narval_screening_preprocessing_m6/1506925/loop_summary.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or args.csv.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = read_summary(args.csv)

    plot_best_per_mouse(data, output_dir)
    plot_combo_summary(data, output_dir)
    plot_pixels_sigma_heatmap(data, output_dir)
    plot_gsr_effect_at_winner(data, output_dir)
    plot_outliers(data, output_dir)
    plot_runtime_tradeoff(data, output_dir)

    print(f"Figures sauvegardees dans: {output_dir}")


if __name__ == "__main__":
    main()
