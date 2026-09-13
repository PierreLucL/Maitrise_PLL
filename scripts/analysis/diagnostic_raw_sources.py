### Diagnostic des fichiers GCaMP recus.
### Ici on veut savoir si le signal arrive deja offsette/centre/croche avant de blamer CURBD.

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

### Permet de lancer le script direct avec `python scripts/...` sans pogner une vieille install pip.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

### Evite les warnings/latences Matplotlib quand le cache user n'est pas writable.
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.environ.get("SLURM_TMPDIR", os.environ.get("TMPDIR", "/tmp")),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from maitrise_curbd.io import load_dataset
from maitrise_curbd.masks import (
    remove_thin_label_artifacts,
    reduce_atlas_to_6_regions,
    subdivide_mask_by_spatial_clustering,
)
from maitrise_curbd.timeseries import extract_timeseries_du_tenseur


DEFAULT_DATASETS = [
    ### Les deux paires controle qui racontent l'histoire des fits CURBD.
    (3, 6, 316),
    (3, 6, 322),
    (9, 6, 415),
    (9, 6, 410),
]

RAW_NEAR_ZERO_THRESHOLDS = [
    ### Si le signal vit proche de zero, ca aide a comprendre son offset et son scaling.
    1e-1,
    1e-2,
    1e-3,
]


def parse_dataset(value):
    ### Format CLI compact: C,M,mouse. Exemple: 9,6,415.
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Chaque dataset doit etre au format cohort,month,mouse."
        )
    try:
        return tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "cohort, month et mouse doivent etre des entiers."
        ) from exc


def build_parser():
    ### Toute la config CLI au meme endroit: local ou Narval, meme affaire.
    now = datetime.now().strftime("%Y-%m-%d_%Hh%M")

    parser = argparse.ArgumentParser(
        description="Diagnostique les fichiers GCaMP tels qu'ils sont recus."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Racine des donnees. Si absent, utilise MAITRISE_DATA_DIR puis data/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "diagnostics" / "raw_sources" / f"run_du_{now}",
        help="Dossier de sortie pour les CSV et figures.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset,
        help=(
            "Dataset au format cohort,month,mouse. "
            "Peut etre repete. Defaut: 316/322/415/410."
        ),
    )
    parser.add_argument(
        "--n-pixels",
        type=int,
        default=100,
        help="Taille cible des sous-regions.",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=12.0,
        help="Frequence d'acquisition en Hz pour annoter les figures.",
    )
    parser.add_argument(
        "--worst-regions-csv",
        type=Path,
        default=None,
        help="CSV optionnel de regions suspectes a zoomer.",
    )
    parser.add_argument(
        "--top-per-mouse",
        type=int,
        default=5,
        help="Nombre de regions F0 problematiques a inspecter par souris.",
    )
    parser.add_argument(
        "--plot-top",
        type=int,
        default=3,
        help="Nombre de regions problematiques a figurer par souris.",
    )
    parser.add_argument(
        "--window-around-sec",
        type=float,
        default=30.0,
        help="Fenetre autour du temps F0 problematique pour les figures region.",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=120,
        help="Nombre de frames par chunk pour les stats exactes sans exploser la RAM.",
    )
    parser.add_argument(
        "--sample-time-step",
        type=int,
        default=10,
        help="Sous-echantillonnage temporel pour les quantiles/histogrammes raw.",
    )
    parser.add_argument(
        "--sample-space-step",
        type=int,
        default=4,
        help="Sous-echantillonnage spatial pour les quantiles/histogrammes raw.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip les datasets deja presents dans raw_source_summary.csv.",
    )
    return parser


def safe_values(x):
    ### Petit filtre finite pour les arrays deja petits. Pour les gros films, on chunk ailleurs.
    values = np.asarray(x, dtype=float).ravel()
    return values[np.isfinite(values)]


def safe_stats(prefix, x):
    ### Stats compactes pour un vecteur/array raisonnable.
    values = safe_values(x)
    if values.size == 0:
        return {
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p01": np.nan,
            f"{prefix}_p05": np.nan,
            f"{prefix}_p95": np.nan,
            f"{prefix}_p99": np.nan,
            f"{prefix}_fraction_negative": np.nan,
            f"{prefix}_fraction_zero": np.nan,
        }

    return {
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p01": float(np.percentile(values, 1)),
        f"{prefix}_p05": float(np.percentile(values, 5)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_p99": float(np.percentile(values, 99)),
        f"{prefix}_fraction_negative": float(np.mean(values < 0)),
        f"{prefix}_fraction_zero": float(np.mean(values == 0)),
    }


def threshold_label(threshold):
    ### Nom de colonne stable: 1e-03 -> 1em03.
    return f"{threshold:.0e}".replace("-", "m")


def exact_movie_counts(gcamp, chunk_frames):
    ### Stats exactes en chunks: plus lent qu'un flatten brutal, mais pas suicidaire pour la RAM.
    gcamp = np.asarray(gcamp)
    total_count = 0
    finite_count = 0
    negative_count = 0
    zero_count = 0
    near_zero_counts = {threshold: 0 for threshold in RAW_NEAR_ZERO_THRESHOLDS}
    running_sum = 0.0
    running_min = np.inf
    running_max = -np.inf

    for start in range(0, gcamp.shape[0], chunk_frames):
        chunk = np.asarray(gcamp[start:start + chunk_frames], dtype=float)
        finite = np.isfinite(chunk)
        values = chunk[finite]

        total_count += chunk.size
        finite_count += values.size
        if values.size == 0:
            continue

        running_sum += float(np.sum(values))
        running_min = min(running_min, float(np.min(values)))
        running_max = max(running_max, float(np.max(values)))
        negative_count += int(np.sum(values < 0))
        zero_count += int(np.sum(values == 0))

        abs_values = np.abs(values)
        for threshold in RAW_NEAR_ZERO_THRESHOLDS:
            near_zero_counts[threshold] += int(np.sum(abs_values < threshold))

    row = {
        "raw_exact_total_count": total_count,
        "raw_exact_finite_count": finite_count,
        "raw_exact_fraction_finite": finite_count / total_count if total_count else np.nan,
        "raw_exact_min": running_min if finite_count else np.nan,
        "raw_exact_max": running_max if finite_count else np.nan,
        "raw_exact_mean": running_sum / finite_count if finite_count else np.nan,
        "raw_exact_fraction_negative": negative_count / finite_count if finite_count else np.nan,
        "raw_exact_fraction_zero": zero_count / finite_count if finite_count else np.nan,
    }

    for threshold, count in near_zero_counts.items():
        row[f"raw_exact_fraction_abs_lt_{threshold_label(threshold)}"] = (
            count / finite_count if finite_count else np.nan
        )

    return row


def sampled_movie_values(gcamp, time_step, space_step):
    ### Echantillon deterministe pour quantiles/histogrammes. Assez leger, assez parlant.
    sample = np.asarray(
        gcamp[::time_step, ::space_step, ::space_step],
        dtype=float,
    )
    return safe_values(sample)


def compute_global_traces(gcamp):
    ### Trace globale et fraction negative par frame: ca expose les offsets et acquisitions bizarres.
    gcamp = np.asarray(gcamp, dtype=float)
    global_mean = np.nanmean(gcamp, axis=(1, 2))
    global_median = np.nanmedian(gcamp, axis=(1, 2))
    fraction_negative = np.nanmean(gcamp < 0, axis=(1, 2))
    return global_mean, global_median, fraction_negative


def build_spatial_pipeline(gcamp, atlas, roi_mask, n_pixels):
    ### Meme segmentation que les diagnostics F0/CURBD pour que les indices de regions matchent.
    clean_atlas = remove_thin_label_artifacts(
        np.asarray(atlas),
        size=5,
        min_fraction=0.25,
    )
    atlas_6 = reduce_atlas_to_6_regions(
        atlas=clean_atlas,
        roi_mask=np.asarray(roi_mask),
    )
    masque_sub, _ = subdivide_mask_by_spatial_clustering(
        atlas_6,
        target_size=n_pixels,
    )
    labels = np.unique(masque_sub[np.isfinite(masque_sub)])
    ts_raw = extract_timeseries_du_tenseur(np.asarray(gcamp), masque_sub)
    return masque_sub, labels, np.asarray(ts_raw, dtype=float)


def summarize_dataset(cohort, month, mouse, gcamp, ts_raw, global_mean, global_median, frac_neg, args):
    ### Une ligne par souris: est-ce que le fichier source a l'air physiquement sain?
    sampled_values = sampled_movie_values(
        gcamp,
        time_step=args.sample_time_step,
        space_step=args.sample_space_step,
    )

    row = {
        "cohort": cohort,
        "month": month,
        "mouse": mouse,
        "shape": "x".join(str(v) for v in np.asarray(gcamp).shape),
        "dtype": str(np.asarray(gcamp).dtype),
        "n_regions": ts_raw.shape[0],
        "n_timepoints": ts_raw.shape[1],
        "sample_time_step": args.sample_time_step,
        "sample_space_step": args.sample_space_step,
    }

    row.update(exact_movie_counts(gcamp, args.chunk_frames))
    row.update(safe_stats("raw_sample", sampled_values))
    row.update(safe_stats("region_mean_raw", ts_raw))
    row.update(safe_stats("global_mean_trace", global_mean))
    row.update(safe_stats("global_median_trace", global_median))
    row.update(safe_stats("frame_fraction_negative", frac_neg))

    if global_mean.size:
        ### Drift rough: est-ce que l'offset global bouge entre debut et fin?
        split = max(1, global_mean.size // 10)
        row["global_mean_first10pct"] = float(np.nanmean(global_mean[:split]))
        row["global_mean_last10pct"] = float(np.nanmean(global_mean[-split:]))
        row["global_mean_last_minus_first"] = (
            row["global_mean_last10pct"] - row["global_mean_first10pct"]
        )

    return row, sampled_values


def load_worst_regions(path, datasets, top_per_mouse):
    ### Charge des regions suspectes. Si absent, le diagnostic reste utile quand meme.
    if path is None or not Path(path).is_file():
        return pd.DataFrame()

    df = pd.read_csv(path)
    wanted = pd.DataFrame(
        [{"cohort": c, "month": m, "mouse": s} for c, m, s in datasets]
    )
    df = df.merge(wanted, on=["cohort", "month", "mouse"], how="inner")
    df = df.sort_values(["cohort", "month", "mouse", "rank"])
    return df.groupby(["cohort", "month", "mouse"], as_index=False).head(top_per_mouse)


def inspect_breaking_regions(cohort, month, mouse, gcamp, masque_sub, labels, ts_raw, worst_df):
    ### Compare les regions qui cassent: raw region moyen + pixels sources sous ce label.
    if worst_df.empty:
        return []

    rows = []
    subset = worst_df[
        (worst_df["cohort"] == cohort)
        & (worst_df["month"] == month)
        & (worst_df["mouse"] == mouse)
    ]

    for worst in subset.itertuples(index=False):
        region_idx = int(worst.region_idx)
        if region_idx < 0 or region_idx >= len(labels):
            continue

        region_label = labels[region_idx]
        pixel_mask = masque_sub == region_label
        region_pixels = np.asarray(gcamp[:, pixel_mask], dtype=float)
        region_trace = ts_raw[region_idx]

        row = {
            "cohort": cohort,
            "month": month,
            "mouse": mouse,
            "rank": int(worst.rank),
            "region_idx": region_idx,
            "region_label": float(region_label),
            "n_pixels_region": int(np.sum(pixel_mask)),
            "time_sec_suspect": getattr(worst, "time_sec_min_abs_F0", np.nan),
            "source_metric_1": getattr(worst, "min_abs_F0", np.nan),
            "source_metric_2": getattr(worst, "max_abs_dff", np.nan),
            "raw_at_suspect_time": getattr(worst, "raw_at_min_abs_F0", np.nan),
        }
        row.update(safe_stats("region_trace_raw", region_trace))
        row.update(safe_stats("region_pixels_raw", region_pixels))
        rows.append(row)

    return rows


def plot_dataset_summary(save_dir, cohort, month, mouse, sampled_values, global_mean, global_median, frac_neg, fs):
    ### Figure rapide par souris: distribution raw + offset global + negatif par frame.
    time = np.arange(global_mean.size) / fs

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)

    axes[0].hist(sampled_values, bins=150, color="tab:blue", alpha=0.85)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title(f"C{cohort} M{month} souris {mouse} | distribution raw sample")
    axes[0].set_xlabel("Raw GCaMP")
    axes[0].set_ylabel("Compte")

    axes[1].plot(time, global_mean, label="mean", linewidth=1.1)
    axes[1].plot(time, global_median, label="median", linewidth=1.1)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Offset global")
    axes[1].legend(loc="best")

    axes[2].plot(time, frac_neg, color="tab:red", linewidth=1.1)
    axes[2].set_xlabel("Temps (s)")
    axes[2].set_ylabel("Fraction negative")
    axes[2].set_ylim(-0.02, 1.02)

    fig.savefig(save_dir / f"C{cohort}_M{month}_mouse{mouse}_raw_source_summary.png", dpi=300)
    plt.close(fig)


def plot_breaking_region(save_dir, cohort, month, mouse, row, ts_raw, fs, window_around_sec):
    ### Zoom sur une region suspecte: est-ce deja weird dans le signal recu?
    region_idx = int(row["region_idx"])
    center_sec = row.get("time_sec_suspect", np.nan)
    if not np.isfinite(center_sec):
        return

    center = int(round(center_sec * fs))
    half_window = int(round(window_around_sec * fs))
    start = max(0, center - half_window)
    stop = min(ts_raw.shape[1], center + half_window + 1)
    time = np.arange(start, stop) / fs

    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.plot(time, ts_raw[region_idx, start:stop], linewidth=1.2)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(center_sec, color="black", linestyle="--", alpha=0.6)
    ax.set_title(
        f"C{cohort} M{month} souris {mouse} | raw region {region_idx} | rank {int(row['rank'])}"
    )
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Raw moyen region")
    fig.savefig(
        save_dir / f"C{cohort}_M{month}_mouse{mouse}_rank{int(row['rank']):02d}_region{region_idx}_raw_zoom.png",
        dpi=300,
    )
    plt.close(fig)


def completed_dataset_keys(summary_csv):
    ### Resume: si une souris est deja resumee, on ne refait pas tout le film.
    if not summary_csv.exists():
        return set()
    df = pd.read_csv(summary_csv)
    required = {"cohort", "month", "mouse"}
    if not required.issubset(df.columns):
        return set()
    return {
        (int(row.cohort), int(row.month), int(row.mouse))
        for row in df.itertuples(index=False)
    }


def append_rows(csv_path, rows):
    ### Sauvegarde progressive. Si ca plante, au moins le travail deja fait reste sur disque.
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        new_df = pd.concat([old_df, new_df], ignore_index=True)
    new_df.to_csv(csv_path, index=False)


def main():
    ### Main lineaire: source -> segmentation -> stats raw -> zoom regions cassees.
    args = build_parser().parse_args()
    datasets = args.dataset if args.dataset else DEFAULT_DATASETS

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = args.output_dir / "raw_source_summary.csv"
    breaking_csv = args.output_dir / "raw_breaking_regions.csv"

    completed = completed_dataset_keys(summary_csv) if args.resume else set()
    worst_df = load_worst_regions(
        args.worst_regions_csv,
        datasets=datasets,
        top_per_mouse=args.top_per_mouse,
    )

    if args.worst_regions_csv:
        print(f"Regions suspectes utilisees : {args.worst_regions_csv}")
    else:
        print("Aucun CSV de regions suspectes trouve: diagnostic source seulement.")

    for cohort, month, mouse in datasets:
        key = (cohort, month, mouse)
        if key in completed:
            print(f"Skip deja termine: C{cohort} M{month} souris {mouse}")
            continue

        print("\n" + "=" * 100)
        print(f"Diagnostic raw source | C{cohort} | M{month} | souris {mouse}")
        print("=" * 100)

        gcamp, atlas, roi_mask = load_dataset(
            cohort=cohort,
            month=month,
            mouse=mouse,
            data_root=args.data_dir,
        )

        global_mean, global_median, frac_neg = compute_global_traces(gcamp)
        masque_sub, labels, ts_raw = build_spatial_pipeline(
            gcamp,
            atlas,
            roi_mask,
            n_pixels=args.n_pixels,
        )

        summary_row, sampled_values = summarize_dataset(
            cohort,
            month,
            mouse,
            gcamp,
            ts_raw,
            global_mean,
            global_median,
            frac_neg,
            args,
        )
        append_rows(summary_csv, [summary_row])

        plot_dataset_summary(
            figures_dir,
            cohort,
            month,
            mouse,
            sampled_values,
            global_mean,
            global_median,
            frac_neg,
            fs=args.fs,
        )

        breaking_rows = inspect_breaking_regions(
            cohort,
            month,
            mouse,
            gcamp,
            masque_sub,
            labels,
            ts_raw,
            worst_df,
        )
        append_rows(breaking_csv, breaking_rows)

        for row in breaking_rows[: args.plot_top]:
            plot_breaking_region(
                figures_dir,
                cohort,
                month,
                mouse,
                row,
                ts_raw,
                fs=args.fs,
                window_around_sec=args.window_around_sec,
            )

        print(
            "Resume rapide: "
            f"raw min={summary_row['raw_exact_min']:.3g}, "
            f"raw max={summary_row['raw_exact_max']:.3g}, "
            f"frac neg={summary_row['raw_exact_fraction_negative']:.3g}, "
            f"global drift={summary_row.get('global_mean_last_minus_first', np.nan):.3g}"
        )

    if summary_csv.exists():
        summary_df = pd.read_csv(summary_csv)
        print("\n" + "=" * 100)
        print("RÉSUMÉ RAW SOURCE")
        print("=" * 100)
        cols = [
            "cohort",
            "month",
            "mouse",
            "raw_exact_min",
            "raw_exact_max",
            "raw_exact_mean",
            "raw_exact_fraction_negative",
            "raw_exact_fraction_abs_lt_1em01",
            "raw_sample_median",
            "global_mean_trace_median",
            "frame_fraction_negative_median",
            "global_mean_last_minus_first",
        ]
        cols = [col for col in cols if col in summary_df.columns]
        print(summary_df[cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("DIAGNOSTIC RAW SOURCE TERMINÉ")
    print(f"CSV résumé : {summary_csv}")
    print(f"CSV régions qui cassent : {breaking_csv}")
    print(f"Figures : {figures_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
