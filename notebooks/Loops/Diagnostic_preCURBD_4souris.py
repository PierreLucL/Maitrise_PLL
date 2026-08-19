
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from maitrise_curbd.io import load_dataset
from maitrise_curbd.masks import (
    remove_thin_label_artifacts,
    reduce_atlas_to_6_regions,
    subdivide_mask_by_spatial_clustering,
)
from maitrise_curbd.timeseries import (
    compute_dff,
    extract_timeseries_du_tenseur,
    regress_out_global_signal,
    smooth_timeseries,
    compute_mean_psd,
    compute_mean_autocorrelation,
    estimate_autocorrelation_timescale,
    compute_cumulative_power_frequencies,
)

datasets = [
    (3, 6, 316),
    (3, 6, 322),
    (9, 6, 415),
    (9, 6, 410),
]

pvar_curbd_reference = {
    316: 0.234,
    322: 0.442,
    415: 0.052,
    410: 0.397,
}

n_pixels = 100
dtData = 1 / 12
fs = 1 / dtData
window_sec_dff = 60
percentile_dff = 8
lissage_sigma = 2
max_lag_seconds = 5.0

titre_du_test = "Diagnostic_preCURBD_4souris"
now = datetime.now()
maintenant = now.strftime("%Y-%m-%d_%Hh%M")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path(titre_du_test) / f"run_du_{maintenant}",
)
args = parser.parse_args()

save_dir = args.output_dir
save_dir.mkdir(parents=True, exist_ok=True)
results_csv = save_dir / "diagnostic_preCURBD_summary.csv"


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else np.nan


def safe_median(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan


def robust_amplitude_per_trace(ts):
    return (
        np.nanpercentile(ts, 95, axis=1)
        - np.nanpercentile(ts, 5, axis=1)
    )


def rms_per_trace(ts):
    return np.sqrt(np.nanmean(ts ** 2, axis=1))


def mean_pairwise_correlation(ts):
    corr = np.corrcoef(ts)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    values = corr[mask]
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else np.nan


def median_pairwise_correlation(ts):
    corr = np.corrcoef(ts)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    values = corr[mask]
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else np.nan


def persistence_pvar_one_frame(ts):
    y_true = ts[:, 1:]
    y_pred = ts[:, :-1]
    ss_res = np.nansum((y_true - y_pred) ** 2)
    y_mean = np.nanmean(y_true)
    ss_tot = np.nansum((y_true - y_mean) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def global_signal_variance(ts):
    return float(np.nanvar(np.nanmean(ts, axis=0)))


def gsr_variance_removed_fraction(ts_before, ts_after):
    var_before = np.nanvar(ts_before, axis=1)
    var_after = np.nanvar(ts_after, axis=1)
    valid = np.isfinite(var_before) & np.isfinite(var_after) & (var_before > 1e-12)
    frac = np.full(ts_before.shape[0], np.nan, dtype=float)
    frac[valid] = 1 - var_after[valid] / var_before[valid]
    return frac


def compute_stage_metrics(ts, stage, cohort, month, mouse, n_subregions, pvar_curbd):
    ts = np.asarray(ts, dtype=float)

    var_trace = np.nanvar(ts, axis=1)
    rms_trace = rms_per_trace(ts)
    amp_trace = robust_amplitude_per_trace(ts)

    lag_times, mean_acf, _ = compute_mean_autocorrelation(
        ts,
        dt=dtData,
        max_lag_seconds=max_lag_seconds,
    )
    tau_acf = estimate_autocorrelation_timescale(lag_times, mean_acf)

    freqs, mean_psd, _ = compute_mean_psd(ts, fs=fs)
    power_freqs, cumulative_fraction = compute_cumulative_power_frequencies(
        freqs,
        mean_psd,
    )

    total_power = np.trapezoid(mean_psd, freqs)
    mask_above_1hz = freqs >= 1
    power_above_1hz = np.trapezoid(
        mean_psd[mask_above_1hz],
        freqs[mask_above_1hz],
    )
    fraction_power_above_1hz = (
        power_above_1hz / total_power
        if total_power > 0
        else np.nan
    )

    row = {
        "cohort": cohort,
        "month": month,
        "mouse": mouse,
        "pVar_CURBD_reference": pvar_curbd,
        "stage": stage,
        "n_pixels": n_pixels,
        "n_subregions": n_subregions,
        "variance_trace_mean": safe_mean(var_trace),
        "variance_trace_median": safe_median(var_trace),
        "rms_trace_mean": safe_mean(rms_trace),
        "rms_trace_median": safe_median(rms_trace),
        "amplitude_P95_P5_mean": safe_mean(amp_trace),
        "amplitude_P95_P5_median": safe_median(amp_trace),
        "mean_pairwise_corr": mean_pairwise_correlation(ts),
        "median_pairwise_corr": median_pairwise_correlation(ts),
        "global_signal_variance": global_signal_variance(ts),
        "tau_acf_seconds": tau_acf,
        "f50_Hz": power_freqs["f50"],
        "f90_Hz": power_freqs["f90"],
        "f95_Hz": power_freqs["f95"],
        "f99_Hz": power_freqs["f99"],
        "fraction_power_above_1Hz": fraction_power_above_1hz,
        "pVar_persistence_1frame": persistence_pvar_one_frame(ts),
        "min_value": float(np.nanmin(ts)),
        "max_value": float(np.nanmax(ts)),
    }

    return row, {
        "lag_times": lag_times,
        "mean_acf": mean_acf,
        "freqs": freqs,
        "mean_psd": mean_psd,
        "cumulative_fraction": cumulative_fraction,
    }


rows = []
all_curves = {}

for cohort, month, mouse in datasets:
    print("\n" + "=" * 100)
    print(f"C{cohort} | M{month} | souris {mouse}")
    print("=" * 100)

    gcamp, atlas, roi_mask = load_dataset(
        cohort=cohort,
        month=month,
        mouse=mouse,
    )

    gcamp = np.asarray(gcamp)
    atlas = np.asarray(atlas)
    roi_mask = np.asarray(roi_mask)

    clean_atlas = remove_thin_label_artifacts(
        atlas,
        size=5,
        min_fraction=0.25,
    )

    atlas_6 = reduce_atlas_to_6_regions(
        atlas=clean_atlas,
        roi_mask=roi_mask,
    )

    masque_sub, info_masque_sub = subdivide_mask_by_spatial_clustering(
        atlas_6,
        target_size=n_pixels,
    )

    labels_valides = np.unique(
        masque_sub[np.isfinite(masque_sub)]
    )
    n_subregions = len(labels_valides)

    print(f"Nombre de sous-régions : {n_subregions}")

    ts_raw = extract_timeseries_du_tenseur(gcamp, masque_sub)
    ts_raw = np.asarray(ts_raw, dtype=float)

    row_raw, curves_raw = compute_stage_metrics(
        ts_raw, "raw", cohort, month, mouse, n_subregions,
        pvar_curbd_reference.get(mouse, np.nan),
    )
    rows.append(row_raw)
    all_curves[(mouse, "raw")] = curves_raw

    ts_dff = compute_dff(
        ts_raw,
        fs=fs,
        window_sec=window_sec_dff,
        percentile=percentile_dff,
    )
    row_dff, curves_dff = compute_stage_metrics(
        ts_dff, "dff", cohort, month, mouse, n_subregions,
        pvar_curbd_reference.get(mouse, np.nan),
    )
    rows.append(row_dff)
    all_curves[(mouse, "dff")] = curves_dff

    ts_gsr = regress_out_global_signal(ts_dff)
    row_gsr, curves_gsr = compute_stage_metrics(
        ts_gsr, "dff_gsr", cohort, month, mouse, n_subregions,
        pvar_curbd_reference.get(mouse, np.nan),
    )

    gsr_removed = gsr_variance_removed_fraction(ts_dff, ts_gsr)
    row_gsr["gsr_variance_removed_mean"] = safe_mean(gsr_removed)
    row_gsr["gsr_variance_removed_median"] = safe_median(gsr_removed)

    rows.append(row_gsr)
    all_curves[(mouse, "dff_gsr")] = curves_gsr

    ts_final = smooth_timeseries(ts_gsr, sigma=lissage_sigma)
    row_final, curves_final = compute_stage_metrics(
        ts_final, "final_sigma2", cohort, month, mouse, n_subregions,
        pvar_curbd_reference.get(mouse, np.nan),
    )

    scale = np.max(ts_final)
    scaled = ts_final / scale
    row_final["fraction_clip_pos_CURBD"] = float(np.mean(scaled > 0.999))
    row_final["fraction_clip_neg_CURBD"] = float(np.mean(scaled < -0.999))

    rows.append(row_final)
    all_curves[(mouse, "final_sigma2")] = curves_final

    pd.DataFrame(rows).to_csv(results_csv, index=False)


df = pd.DataFrame(rows)
df.to_csv(results_csv, index=False)

final_df = df[df["stage"] == "final_sigma2"].copy()

print("\n" + "=" * 100)
print("RÉSUMÉ FINAL — SIGNAL DONNÉ À CURBD")
print("=" * 100)

cols = [
    "cohort",
    "mouse",
    "pVar_CURBD_reference",
    "n_subregions",
    "variance_trace_median",
    "amplitude_P95_P5_median",
    "mean_pairwise_corr",
    "global_signal_variance",
    "tau_acf_seconds",
    "f50_Hz",
    "f90_Hz",
    "f95_Hz",
    "f99_Hz",
    "fraction_power_above_1Hz",
    "pVar_persistence_1frame",
]

print(final_df[cols].to_string(index=False))

gsr_df = df[df["stage"] == "dff_gsr"].copy()

print("\n" + "=" * 100)
print("EFFET DE LA RÉGRESSION DU SIGNAL GLOBAL")
print("=" * 100)

gsr_cols = [
    "cohort",
    "mouse",
    "pVar_CURBD_reference",
    "gsr_variance_removed_mean",
    "gsr_variance_removed_median",
    "mean_pairwise_corr",
    "global_signal_variance",
]

print(gsr_df[gsr_cols].to_string(index=False))


plt.figure(figsize=(9, 6))
for cohort, month, mouse in datasets:
    curves = all_curves[(mouse, "final_sigma2")]
    label = f"{mouse} (pVar={pvar_curbd_reference.get(mouse, np.nan):.3f})"
    plt.plot(curves["lag_times"], curves["mean_acf"], label=label)
plt.axhline(np.exp(-1), linestyle="--", alpha=0.5)
plt.xlabel("Lag (s)")
plt.ylabel("Autocorrélation moyenne")
plt.title("Autocorrélation — signal final donné à CURBD")
plt.legend()
plt.tight_layout()
plt.savefig(save_dir / "autocorrelation_finale_4souris.png", dpi=300)
plt.close()


plt.figure(figsize=(9, 6))
for cohort, month, mouse in datasets:
    curves = all_curves[(mouse, "final_sigma2")]
    label = f"{mouse} (pVar={pvar_curbd_reference.get(mouse, np.nan):.3f})"
    plt.plot(curves["freqs"], curves["mean_psd"], label=label)
plt.yscale("log")
plt.xlim(0, 2)
plt.xlabel("Fréquence (Hz)")
plt.ylabel("PSD moyenne")
plt.title("PSD — signal final donné à CURBD")
plt.legend()
plt.tight_layout()
plt.savefig(save_dir / "PSD_finale_4souris.png", dpi=300)
plt.close()


plt.figure(figsize=(7, 5))
plt.scatter(final_df["tau_acf_seconds"], final_df["pVar_CURBD_reference"])
for _, row in final_df.iterrows():
    plt.annotate(
        str(int(row["mouse"])),
        (row["tau_acf_seconds"], row["pVar_CURBD_reference"]),
    )
plt.xlabel("tau ACF (s)")
plt.ylabel("pVar CURBD")
plt.title("pVar CURBD vs échelle temporelle")
plt.tight_layout()
plt.savefig(save_dir / "pVar_vs_tauACF.png", dpi=300)
plt.close()


plt.figure(figsize=(7, 5))
plt.scatter(final_df["mean_pairwise_corr"], final_df["pVar_CURBD_reference"])
for _, row in final_df.iterrows():
    plt.annotate(
        str(int(row["mouse"])),
        (row["mean_pairwise_corr"], row["pVar_CURBD_reference"]),
    )
plt.xlabel("Corrélation moyenne inter-sous-régions")
plt.ylabel("pVar CURBD")
plt.title("pVar CURBD vs corrélation moyenne")
plt.tight_layout()
plt.savefig(save_dir / "pVar_vs_corr_moyenne.png", dpi=300)
plt.close()

print("\n" + "=" * 100)
print("DIAGNOSTIC TERMINÉ")
print(f"CSV : {results_csv}")
print(f"Figures : {save_dir}")
print("=" * 100)
