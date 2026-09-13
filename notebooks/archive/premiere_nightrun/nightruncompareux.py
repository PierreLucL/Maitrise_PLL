import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_boolean_effect(df, bool_col="use_dff", y_col="pVarmax"):

    means = df.groupby(bool_col)[y_col].mean()
    stds = df.groupby(bool_col)[y_col].std()

    fig, ax = plt.subplots(figsize=(5,4))

    ax.bar(
        ["False", "True"],
        means.values,
        yerr=stds.values,
        capsize=5
    )

    ax.set_ylabel(y_col)
    ax.set_title(f"Effet de {bool_col}")
    plt.tight_layout()
    plt.show()

    return fig, ax

def plot_pvarmax_from_night_csv(
    csv_path="night_run_results.csv",
    y_col="pVarmax",
    group_col="use_global_regression",
    x_col="lissage_sigma",
    hue_col="n_pixels",
    aggregate="mean",
    figsize=(9, 5),
    title=None,
    save_path=None,
):
    """
    Plot pVarmax en fonction d'un paramètre, avec moyenne par condition.

    Exemples:
    - x_col="lissage_sigma", group_col="use_global_regression", hue_col="n_pixels"
    - x_col="n_pixels", group_col="use_global_regression", hue_col="lissage_sigma"
    """

    df = pd.read_csv(csv_path)

    # --- tolérance aux variantes de noms ---
    aliases = {
        "pVarmax": ["pVarmax", "pvarmax", "pVar_max", "pVarMax", "max_pVar"],
        "use_global_regression": ["use_global_regression", "global_regression", "global_reg", "use_global_reg"],
        "lissage_sigma": ["lissage_sigma", "sigma", "smoothing_sigma"],
        "n_pixels": ["n_pixels", "nb_pixels", "num_pixels"],
    }

    def resolve_col(name):
        if name in df.columns:
            return name
        for canonical, options in aliases.items():
            if name == canonical:
                for opt in options:
                    if opt in df.columns:
                        return opt
        raise ValueError(f"Colonne introuvable: {name}. Colonnes disponibles: {list(df.columns)}")

    y_col = resolve_col(y_col)
    group_col = resolve_col(group_col)
    x_col = resolve_col(x_col)
    hue_col = resolve_col(hue_col) if hue_col is not None else None

    # Nettoyage minimal
    df = df.copy()
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[y_col, x_col, group_col])

    # Agrégation
    groupby_cols = [x_col, group_col]
    if hue_col is not None:
        groupby_cols.append(hue_col)

    if aggregate == "mean":
        agg_df = df.groupby(groupby_cols, as_index=False)[y_col].mean()
    elif aggregate == "median":
        agg_df = df.groupby(groupby_cols, as_index=False)[y_col].median()
    else:
        raise ValueError("aggregate doit être 'mean' ou 'median'")

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    if hue_col is None:
        for group_value, sub in agg_df.groupby(group_col):
            sub = sub.sort_values(x_col)
            ax.plot(
                sub[x_col],
                sub[y_col],
                marker="o",
                label=f"{group_col}={group_value}",
            )
    else:
        for (group_value, hue_value), sub in agg_df.groupby([group_col, hue_col]):
            sub = sub.sort_values(x_col)
            ax.plot(
                sub[x_col],
                sub[y_col],
                marker="o",
                label=f"{group_col}={group_value}, {hue_col}={hue_value}",
            )

    ax.set_xlabel(x_col)
    ax.set_ylabel(f"{aggregate} {y_col}")
    ax.set_title(title or f"{aggregate} {y_col} en fonction de {x_col}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax, agg_df


# 1. Effet du lissage sigma, séparé par global regression et n_pixels
fig, ax, table = plot_pvarmax_from_night_csv(
    x_col="lissage_sigma",
    group_col="use_global_regression",
    hue_col="n_pixels"
)
plt.show()

# 2. Effet du nombre de pixels, séparé par global regression et sigma
fig, ax, table = plot_pvarmax_from_night_csv(
    x_col="n_pixels",
    group_col="use_global_regression",
    hue_col="lissage_sigma")

plt.show()

# 3. Juste global regression True vs False, moyenné sur tout le reste
fig, ax, table = plot_pvarmax_from_night_csv(
    x_col="n_pixels",
    group_col="use_dff",
    hue_col="lissage_sigma")

plt.show()