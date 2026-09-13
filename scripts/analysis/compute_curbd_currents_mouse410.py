"""Calcule et compare les courants CURBD pour les pkl de la souris 410."""

import argparse
import csv
import hashlib
import pickle
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.gridspec import GridSpec


### Petit setup pour que le script marche direct dans VSCode sans pip install -e obligatoire.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


REGION_COLORS = {
    0: "#0047AB",
    1: "#FF7F00",
    2: "#00A550",
    3: "#A020F0",
    4: "#E60026",
    5: "#00B7EB",
}


def load_pickle_compatible(path):
    ### Les pkl Narval peuvent mentionner numpy._core; ce shim garde ton Mac relax.
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


def safe_float(value, default=np.nan):
    ### Les vieux csv/pkl ont parfois des strings; on convertit sans faire de drame.
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=-1):
    ### Meme idee, mais pour les entiers qui se promenent dans les metadata.
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def short_run_label(data):
    ### Label compact pour les figures: juste assez d'info pour comparer sans roman.
    params = data.get("parameters", {})
    row = data.get("row", {})
    mouse = safe_int(params.get("mouse", row.get("mouse")))
    n_pixels = safe_int(params.get("n_pixels", row.get("n_pixels")))
    n_regions = safe_int(params.get("n_subregions", row.get("n_subregions")))
    n_train = safe_int(params.get("nRunTrain", row.get("nRunTrain")))
    pvar = safe_float(row.get("pVar_finale", row.get("pVar", np.nan)))
    return f"mouse{mouse}_pix{n_pixels}_N{n_regions}_train{n_train}_pVar{pvar:.3f}"


def find_pickles(input_dir, mouse, include_pattern, verbose=False):
    ### Scan large dans results, puis filtre par metadata: plus robuste que se fier au nom de fichier.
    paths = sorted(Path(input_dir).rglob(include_pattern))
    kept = []
    for path in paths:
        try:
            data = load_pickle_compatible(path)
        except Exception as exc:
            if verbose:
                print(f"[skip] {path}: impossible de lire ({exc})")
            continue

        params = data.get("parameters", {})
        row = data.get("row", {})
        this_mouse = safe_int(params.get("mouse", row.get("mouse")))
        if this_mouse != mouse:
            continue

        if data.get("J_final") is None or data.get("RNN_final") is None or data.get("regions") is None:
            if verbose:
                print(f"[skip] {path}: manque J_final/RNN_final/regions")
            continue

        kept.append((path, data))

    kept.sort(key=lambda item: safe_float(item[1].get("row", {}).get("pVar_finale", np.nan)))
    return kept


def compute_total_currents(data, dtype=np.float32):
    """
    Calcule les courants totaux source -> cible sans construire le gros CURBD complet.

    Le CURBD complet ferait, pour chaque paire, une matrice
    (unites cible x temps). Ici on somme les poids vers la region cible avant
    de multiplier par l'activite source. Meme resultat pour le courant total,
    beaucoup moins de RAM qui part en feu.
    """
    J = np.asarray(data["J_final"], dtype=dtype)
    RNN = np.asarray(data["RNN_final"], dtype=dtype)
    regions = np.asarray(data["regions"], dtype=object)

    n_regions = regions.shape[0]
    currents = {}

    for i_target in range(n_regions):
        target_idx = np.asarray(regions[i_target, 1], dtype=int)
        for i_source in range(n_regions):
            source_idx = np.asarray(regions[i_source, 1], dtype=int)

            ### Somme des poids de toutes les unites cible recevant la source.
            summed_weights = J[np.ix_(target_idx, source_idx)].sum(axis=0)
            current = summed_weights @ RNN[source_idx, :]
            currents[(i_target, i_source)] = np.asarray(current, dtype=np.float32)

    return currents


def target_region_size(data, i_target):
    ### Taille de la region cible; utile pour ne pas comparer une grosse region a une mini brute force.
    regions = np.asarray(data["regions"], dtype=object)
    return max(1, len(np.asarray(regions[i_target, 1], dtype=int)))


def normalize_current_for_display(current, data, i_target, mode):
    ### Les courants bruts sont sauvegardes; ici on choisit juste comment les rendre lisibles.
    y = np.asarray(current, dtype=float).copy()

    if mode in {"centered", "centered_sqrt_target", "centered_mean_target"}:
        y = y - np.nanmean(y)

    if mode == "centered_sqrt_target":
        y = y / np.sqrt(target_region_size(data, i_target))
    elif mode == "centered_mean_target":
        y = y / target_region_size(data, i_target)

    return y


def currents_to_array(currents, n_regions):
    ### Format stable: lignes = target/source, colonnes = temps.
    rows = []
    labels = []
    for i_target in range(n_regions):
        for i_source in range(n_regions):
            rows.append(np.asarray(currents[(i_target, i_source)], dtype=np.float32))
            labels.append((i_target, i_source))
    return np.vstack(rows), labels


def gradient_line(x, y, ax, color_start, color_end, lw=0.8):
    ### Meme vibe que ton ancien plot: debut couleur source, fin couleur cible.
    cmap = LinearSegmentedColormap.from_list("source_target", [color_start, color_end])
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidth=lw)
    lc.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(lc)
    return lc


def make_mask_rgb(data):
    ### Certains vieux pkl ont le masque; les nouveaux non. Si absent, on skip clean.
    masque_sub = data.get("masque_sub")
    regions = data.get("regions")
    if masque_sub is None or regions is None:
        return None

    masque_sub = np.asarray(masque_sub)
    regions = np.asarray(regions, dtype=object)
    mask_rgb = np.ones((*masque_sub.shape, 3), dtype=float)

    for i_region in range(regions.shape[0]):
        color = to_rgb(REGION_COLORS.get(i_region, "#777777"))
        for idx in np.asarray(regions[i_region, 1], dtype=int):
            mask_rgb[masque_sub == idx] = color

    if np.any(np.isnan(masque_sub)):
        mask_rgb[np.isnan(masque_sub)] = [1, 1, 1]

    return mask_rgb


def plot_currents_matrix(
    data,
    currents,
    output_path,
    smooth_sigma=0.0,
    max_seconds=None,
    display_mode="centered_sqrt_target",
):
    ### Figure principale: une case par courant cible/source, comme ton ancien notebook.
    regions = np.asarray(data["regions"], dtype=object)
    n_regions = regions.shape[0]
    t_rnn = np.asarray(data.get("tRNN", np.arange(next(iter(currents.values())).size)), dtype=float)

    if max_seconds is not None:
        keep = t_rnn <= (t_rnn[0] + max_seconds)
    else:
        keep = np.ones(t_rnn.shape, dtype=bool)
    t_plot = t_rnn[keep]

    plot_currents = {}
    for key, current in currents.items():
        i_target, _ = key
        y = normalize_current_for_display(
            current,
            data,
            i_target=i_target,
            mode=display_mode,
        )[keep]
        if smooth_sigma and smooth_sigma > 0:
            ### Import local: si scipy manque, le calcul reste possible, juste sans lissage visuel.
            try:
                from scipy.ndimage import gaussian_filter1d

                y = gaussian_filter1d(y, sigma=float(smooth_sigma))
            except Exception:
                pass
        plot_currents[key] = y

    all_values = np.concatenate(list(plot_currents.values()))
    max_abs = np.nanpercentile(np.abs(all_values), 99)
    if not np.isfinite(max_abs) or max_abs <= 1e-12:
        max_abs = 1.0

    mask_rgb = make_mask_rgb(data)
    has_mask = mask_rgb is not None

    fig = plt.figure(figsize=(13.5, 8.8))
    outer = GridSpec(
        1,
        2 if has_mask else 1,
        width_ratios=[1.0, 5.4] if has_mask else [1.0],
        wspace=0.16,
        figure=fig,
    )

    if has_mask:
        ax_mask = fig.add_subplot(outer[0, 0])
        ax_mask.imshow(mask_rgb)
        ax_mask.set_title("Regions", fontsize=10)
        ax_mask.axis("off")
        right_spec = outer[0, 1]
    else:
        right_spec = outer[0, 0]

    grid = right_spec.subgridspec(n_regions, n_regions, wspace=0.08, hspace=0.08)

    for i_target in range(n_regions):
        for i_source in range(n_regions):
            ax = fig.add_subplot(grid[i_target, i_source])
            current = plot_currents[(i_target, i_source)]
            source_color = REGION_COLORS.get(i_source, "#444444")
            target_color = REGION_COLORS.get(i_target, "#444444")

            gradient_line(
                t_plot,
                current,
                ax,
                source_color,
                target_color,
                lw=0.85 if i_source == i_target else 0.55,
            )
            ax.axhline(0, color="black", linewidth=0.4, alpha=0.25)
            ax.set_xlim(t_plot[0], t_plot[-1])
            ax.set_ylim(-max_abs, max_abs)

            if i_target == 0:
                ax.set_title(str(regions[i_source, 0]), fontsize=8, color=source_color)
            if i_source == 0:
                ax.set_ylabel(str(regions[i_target, 0]), fontsize=8, color=target_color)
            if i_target != n_regions - 1:
                ax.set_xticklabels([])
            if i_source != 0:
                ax.set_yticklabels([])

            ax.tick_params(axis="both", labelsize=6, length=2)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_alpha(0.45)

    params = data.get("parameters", {})
    row = data.get("row", {})
    title = (
        "Courants CURBD source -> cible | "
        f"souris {safe_int(params.get('mouse', row.get('mouse')))} | "
        f"n_pixels {safe_int(params.get('n_pixels', row.get('n_pixels')))} | "
        f"N {safe_int(params.get('n_subregions', row.get('n_subregions')))} | "
        f"pVar {safe_float(row.get('pVar_finale', np.nan)):.3f} | "
        f"vue {display_mode}"
    )
    fig.suptitle(title, fontsize=14, y=0.98)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_currents(output_base, data, currents):
    ### Sauvegarde compacte: array 36 x temps + labels, facile a reloader plus tard.
    regions = np.asarray(data["regions"], dtype=object)
    arr, labels = currents_to_array(currents, regions.shape[0])
    t_rnn = np.asarray(data.get("tRNN", np.arange(arr.shape[1])), dtype=float)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_base.with_suffix(".npz"),
        currents=arr,
        labels=np.asarray(labels, dtype=int),
        tRNN=t_rnn,
        region_names=np.asarray([str(r[0]) for r in regions], dtype=object),
    )

    with output_base.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "source", "mean", "std", "rms", "max_abs"])
        for (target, source), current in currents.items():
            y = np.asarray(current, dtype=float)
            writer.writerow(
                [
                    target,
                    source,
                    np.nanmean(y),
                    np.nanstd(y),
                    np.sqrt(np.nanmean(y**2)),
                    np.nanmax(np.abs(y)),
                ]
            )

    return arr, labels


def vector_corr(a, b):
    ### Correlation globale entre deux fingerprints de courants.
    n = min(a.size, b.size)
    if n < 3:
        return np.nan
    aa = np.asarray(a[:n], dtype=float)
    bb = np.asarray(b[:n], dtype=float)
    ok = np.isfinite(aa) & np.isfinite(bb)
    if ok.sum() < 3:
        return np.nan
    aa = aa[ok] - np.nanmean(aa[ok])
    bb = bb[ok] - np.nanmean(bb[ok])
    denom = np.sqrt(np.sum(aa * aa) * np.sum(bb * bb))
    if denom <= 1e-12:
        return np.nan
    return float(np.sum(aa * bb) / denom)


def currents_fingerprint_for_similarity(data, currents, display_mode):
    ### Similarite basee sur la meme version que la figure, sinon les offsets dominent tout.
    regions = np.asarray(data["regions"], dtype=object)
    rows = []
    for i_target in range(regions.shape[0]):
        for i_source in range(regions.shape[0]):
            rows.append(
                normalize_current_for_display(
                    currents[(i_target, i_source)],
                    data,
                    i_target=i_target,
                    mode=display_mode,
                )
            )
    return np.vstack(rows).reshape(-1)


def plot_similarity_matrix(labels, fingerprints, output_dir):
    ### Petit verdict visuel: est-ce que les fingerprints CURBD se ressemblent entre runs?
    n = len(fingerprints)
    sim = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            sim[i, j] = vector_corr(fingerprints[i], fingerprints[j])

    fig, ax = plt.subplots(figsize=(max(7, 0.8 * n + 3), max(6, 0.75 * n + 2)))
    im = ax.imshow(sim, cmap="bwr", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Similarite globale des courants CURBD")

    for i in range(n):
        for j in range(n):
            value = sim[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}" if np.isfinite(value) else "NA",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if np.isfinite(value) and abs(value) > 0.55 else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    fig.savefig(output_dir / "curbd_currents_similarity_mouse410.png", dpi=220)
    fig.savefig(output_dir / "curbd_currents_similarity_mouse410.pdf")
    plt.close(fig)

    with (output_dir / "curbd_currents_similarity_mouse410.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run"] + labels)
        for label, row in zip(labels, sim):
            writer.writerow([label] + list(row))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "curbd_currents_mouse410")
    parser.add_argument("--mouse", type=int, default=410)
    parser.add_argument("--pattern", default="*.pkl")
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument(
        "--display-mode",
        choices=["raw", "centered", "centered_sqrt_target", "centered_mean_target"],
        default="centered_sqrt_target",
        help=(
            "raw garde le courant total; centered retire la moyenne; "
            "centered_sqrt_target retire la moyenne et divise par sqrt(N cible), "
            "ce qui donne habituellement la figure la plus lisible."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    found = find_pickles(
        args.input_dir,
        mouse=args.mouse,
        include_pattern=args.pattern,
        verbose=args.verbose,
    )
    if args.limit is not None:
        found = found[: args.limit]

    if not found:
        raise SystemExit(f"Aucun pkl utilisable trouve pour mouse={args.mouse} dans {args.input_dir}")

    summary_rows = []
    fingerprints = []
    fingerprint_labels = []

    print(f"PKL retenus pour souris {args.mouse}: {len(found)}")
    for idx, (path, data) in enumerate(found, start=1):
        # Identite du contenu : distingue graines, parametres et fichiers remplaces.
        with path.open("rb") as source:
            source_hash = hashlib.file_digest(source, "sha256").hexdigest()
        seed = data.get("parameters", {}).get("seed", "legacy")
        label = f"{short_run_label(data)}_seed{seed}_sha256_{source_hash}"
        run_dir = args.output_dir / label
        output_base = run_dir / "curbd_currents_total"
        figure_base = run_dir / "curbd_currents_matrix"

        if output_base.with_suffix(".npz").exists() and not args.overwrite:
            loaded = np.load(output_base.with_suffix(".npz"), allow_pickle=True)
            arr = loaded["currents"]
            labels_array = loaded["labels"]
            currents = {
                tuple(map(int, labels_array[i])): arr[i]
                for i in range(labels_array.shape[0])
            }
            print(f"[{idx}/{len(found)}] deja calcule: {label}")
        else:
            print(f"[{idx}/{len(found)}] calcul: {label}")
            t0 = time.time()
            currents = compute_total_currents(data)
            arr, _ = save_currents(output_base, data, currents)
            print(f"    fait en {(time.time() - t0) / 60:.1f} min")

        plot_currents_matrix(
            data,
            currents,
            figure_base,
            smooth_sigma=args.smooth_sigma,
            max_seconds=args.max_seconds,
            display_mode=args.display_mode,
        )

        params = data.get("parameters", {})
        row = data.get("row", {})
        summary_rows.append(
            {
                "label": label,
                "pkl": str(path),
                "mouse": safe_int(params.get("mouse", row.get("mouse"))),
                "n_pixels": safe_int(params.get("n_pixels", row.get("n_pixels"))),
                "n_subregions": safe_int(params.get("n_subregions", row.get("n_subregions"))),
                "nRunTrain": safe_int(params.get("nRunTrain", row.get("nRunTrain"))),
                "pVar_finale": safe_float(row.get("pVar_finale", np.nan)),
                "output_npz": str(output_base.with_suffix(".npz")),
                "figure_png": str(figure_base.with_suffix(".png")),
            }
        )
        fingerprints.append(
            currents_fingerprint_for_similarity(
                data,
                currents,
                display_mode=args.display_mode,
            )
        )
        fingerprint_labels.append(label.replace("mouse410_", ""))

    with (args.output_dir / "curbd_currents_mouse410_summary.csv").open("w", newline="") as f:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    plot_similarity_matrix(fingerprint_labels, fingerprints, args.output_dir)

    print()
    print(f"Termine. Resultats dans: {args.output_dir}")
    print(f"Resume: {args.output_dir / 'curbd_currents_mouse410_summary.csv'}")
    print(f"Similarite: {args.output_dir / 'curbd_currents_similarity_mouse410.png'}")


if __name__ == "__main__":
    main()
