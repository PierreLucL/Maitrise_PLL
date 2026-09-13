"""Figures simples des screenings CURBD réalisés avec la souris 409."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_csv(results_dir, relative_path):
    data = pd.read_csv(results_dir / relative_path)
    data = data[(data["mouse"] == 409) & (data["status"] == "done")]
    return data


def save(fig, output_dir, name):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png")
    fig.savefig(output_dir / f"{name}.pdf")
    plt.close(fig)
    print(output_dir / f"{name}.png")


def annotate_points(ax, x, y):
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.3f}", (xi, yi), xytext=(0, 5),
                    textcoords="offset points", ha="center")


def plot_amp(results_dir, output_dir):
    data = read_csv(
        results_dir,
        "FAST_screening_ampInWN_g08_tauRNN033_dtFactor2_sigma2/"
        "run_du_2026-08-18_18h51/night_run_summary.csv",
    ).sort_values("ampInWN")
    fig, ax = plt.subplots()
    ax.plot(data["ampInWN"], data["pVar_finale"], "o-")
    annotate_points(ax, data["ampInWN"], data["pVar_finale"])
    ax.axhline(0, color="black")
    ax.set_xlabel("ampInWN")
    ax.set_ylabel("pVar finale")
    ax.set_title("Souris 409 : pVar en fonction de ampInWN (g = 0.8)")
    save(fig, output_dir, "409_pvar_ampInWN")


def plot_p0(results_dir, output_dir):
    data = read_csv(
        results_dir,
        "FAST_screening_P0_tauRNN033_dtFactor2_sigma2/"
        "run_du_2026-08-18_12h10/night_run_summary.csv",
    ).sort_values("P0")
    fig, ax = plt.subplots()
    ax.plot(data["P0"], data["pVar_finale"], "o-")
    annotate_points(ax, data["P0"], data["pVar_finale"])
    ax.axhline(0, color="black")
    ax.set_xscale("log")
    ax.set_xlabel("P0")
    ax.set_ylabel("pVar finale")
    ax.set_title("Souris 409 : pVar en fonction de P0")
    save(fig, output_dir, "409_pvar_P0")


def plot_g_amp_grid(results_dir, output_dir):
    data = read_csv(
        results_dir,
        "FAST_grid_g_ampInWN_tauRNN033_dtFactor2_sigma2/"
        "run_du_2026-08-18_13h50/night_run_summary.csv",
    )
    pivot = data.pivot(index="g", columns="ampInWN", values="pVar_finale")
    fig, ax = plt.subplots()
    image = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    ax.set_xlabel("ampInWN")
    ax.set_ylabel("g")
    ax.set_title("Souris 409 : pVar pour la grille g × ampInWN")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i, j]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="pVar finale")
    save(fig, output_dir, "409_pvar_grille_g_ampInWN")


def plot_tau_sigma(results_dir, output_dir):
    data = read_csv(
        results_dir,
        "test_tauRNN_sigma_C8_M6_409/run_du_2026-08-06_03h22/"
        "night_run_summary.csv",
    )
    pivot = data.pivot(index="lissage_sigma_frames", columns="tauRNN",
                       values="pVar_finale")
    fig, ax = plt.subplots()
    image = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    ax.set_xlabel("tauRNN")
    ax.set_ylabel("Sigma du lissage (frames)")
    ax.set_title("Souris 409 : grille tauRNN × lissage")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="pVar finale")
    save(fig, output_dir, "409_pvar_grille_tauRNN_sigma")


def plot_tau_dtfactor(results_dir, output_dir):
    data = read_csv(
        results_dir,
        "test_tauRNN_dtFactor_sigma2_CURBD/run_du_2026-08-17_21h14/"
        "night_run_summary.csv",
    )
    fig, ax = plt.subplots()
    for dt_factor, group in data.groupby("dtFactor"):
        group = group.sort_values("tauRNN")
        ax.plot(group["tauRNN"], group["pVar_finale"], "o-",
                label=f"dtFactor = {dt_factor}")
        annotate_points(ax, group["tauRNN"], group["pVar_finale"])
    ax.axhline(0, color="black")
    ax.set_xlabel("tauRNN")
    ax.set_ylabel("pVar finale")
    ax.set_title("Souris 409 : pVar en fonction de tauRNN et dtFactor")
    ax.legend()
    save(fig, output_dir, "409_pvar_tauRNN_dtFactor")


def plot_pixels(results_dir, output_dir):
    path = (
        results_dir.parent / "scripts/FAST_screening_n_pixels_CURBD/"
        "run_du_2026-08-18_20h45/night_run_summary.csv"
    )
    data = pd.read_csv(path)
    data = data[(data["mouse"] == 409) & (data["status"] == "done")]
    data = data.sort_values("n_pixels")
    fig, ax = plt.subplots()
    ax.plot(data["n_pixels"], data["pVar_finale"], "o-")
    annotate_points(ax, data["n_pixels"], data["pVar_finale"])
    ax.axhline(0, color="black")
    ax.set_xlabel("Nombre de pixels par sous-région")
    ax.set_ylabel("pVar finale")
    ax.set_title("Souris 409 : pVar en fonction du nombre de pixels")
    save(fig, output_dir, "409_pvar_n_pixels")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("/Users/pierre-luclarouche/Desktop/Maîtrise/Maitrise_PLL/Results"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures_409"))
    return parser.parse_args()


def main():
    args = parse_args()
    plot_amp(args.results_dir, args.output_dir)
    plot_p0(args.results_dir, args.output_dir)
    plot_g_amp_grid(args.results_dir, args.output_dir)
    plot_tau_sigma(args.results_dir, args.output_dir)
    plot_tau_dtfactor(args.results_dir, args.output_dir)
    plot_pixels(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
