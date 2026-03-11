import pandas as pd
import numpy as np
import scipy.stats as ss
from itables import init_notebook_mode

init_notebook_mode(all_interactive=True)


def marginal_balance_df(df_full, sampled_df, cols):
    rows = []

    for col in cols:
        full = df_full[col].value_counts(normalize=True)
        samp = sampled_df[col].value_counts(normalize=True)

        all_cats = full.index.union(samp.index)

        for cat in all_cats:
            rows.append(
                {
                    "variable": col,
                    "category": cat,
                    "freq_full": full.get(cat, 0.0),
                    "freq_sample": samp.get(cat, 0.0),
                    "abs_diff": abs(full.get(cat, 0.0) - samp.get(cat, 0.0)),
                }
            )

    return pd.DataFrame(rows)


def tvd(p, q):
    idx = p.index.union(q.index)
    return (
        0.5 * (p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).abs().sum()
    )


def normalized_entropy(series):
    p = series.value_counts(normalize=True)
    return ss.entropy(p) / np.log(len(p))


def combination_diversity(df, cols):
    return df[cols].drop_duplicates().shape[0] / len(df)


def joint_entropy(df, cols):
    combos = df[cols].astype(str).agg("_".join, axis=1)
    p = combos.value_counts(normalize=True)
    return ss.entropy(p)


def coverage(df, sampled_df, cols):
    orig = set(map(tuple, df[cols].values))
    samp = set(map(tuple, sampled_df[cols].values))
    coverage = len(orig & samp) / len(orig)
    return coverage


def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(ct)[0]
    n = ct.sum().sum()
    return np.sqrt(chi2 / (n * (min(ct.shape) - 1)))


"""
for i, c1 in enumerate(cols):
    for c2 in cols[i + 1 :]:
        print(
            c1, c2, cramers_v(df[c1], df[c2]), cramers_v(sampled_df[c1], sampled_df[c2])
        )
"""


### Visualization helper functions


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy


def plot_marginal_distributions(df, sampled_df, col, df_name="", ax=None):
    full = df[col].value_counts(normalize=True)
    samp = sampled_df[col].value_counts(normalize=True)

    plot_df = (
        pd.concat([full, samp], axis=1, keys=["Full", "Sample"]).fillna(0).sort_index()
    )

    ax = ax or plt.gca()
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title(f"Marginal distribution: {col}, {df_name}")
    ax.set_ylabel("Proportion")
    ax.legend()


def plot_marginal_distributions_plotly(df, sampled_df, col, df_name=""):
    """Plot marginal distributions (normalized value counts) using Plotly."""
    import pandas as pd
    import plotly.express as px
    import kaleido

    kaleido.start_sync_server(n=5)

    # Normalized value counts
    full = df[col].value_counts(normalize=True)
    samp = sampled_df[col].value_counts(normalize=True)

    # Combine and align categories
    plot_df = (
        pd.concat([full, samp], axis=1, keys=["Full", "Sample"])
        .fillna(0)
        .sort_index()
        .reset_index()
    )

    plot_df.columns = [col, "Full", "Sample"]

    # Convert to long format for Plotly
    plot_df_long = plot_df.melt(
        id_vars=col,
        value_vars=["Full", "Sample"],
        var_name="Dataset",
        value_name="Proportion",
    )

    fig = px.bar(
        plot_df_long,
        x=col,
        y="Proportion",
        color="Dataset",
        barmode="group",
        title=f"Marginal distribution: {col}, {df_name}",
    )

    fig.update_layout(
        xaxis_title=col, yaxis_title="Proportion", template="plotly_white", height=400
    )

    fig.show()


def plot_marginal_abs_diff(df, sampled_df, col, df_name="", ax=None):
    full = df[col].value_counts(normalize=True)
    samp = sampled_df[col].value_counts(normalize=True)

    diff = (full - samp).abs().sort_values(ascending=False)

    ax = ax or plt.gca()
    diff.plot(kind="bar", ax=ax)
    ax.set_title(f"Absolute frequency difference: {col}, {df_name}")
    ax.set_ylabel("|Δ proportion|")


def plot_marginal_abs_diff_plotly(df, sampled_df, cols, df_name=""):
    """Plot absolute difference in normalized value counts using Plotly."""
    import plotly.express as px
    import pandas as pd
    import kaleido

    kaleido.start_sync_server(n=5)

    for col in cols:

        # Normalized value counts
        full = df[col].value_counts(normalize=True)
        samp = sampled_df[col].value_counts(normalize=True)

        # Align indexes to avoid NaNs for missing categories
        combined_index = full.index.union(samp.index)
        full = full.reindex(combined_index, fill_value=0)
        samp = samp.reindex(combined_index, fill_value=0)

        # Absolute difference sorted descending
        diff = (full - samp).abs().sort_values(ascending=False)

        # Convert to DataFrame for Plotly
        diff_df = diff.reset_index()
        diff_df.columns = [col, "|Δ proportion|"]

        fig = px.bar(
            diff_df,
            x=col,
            y="|Δ proportion|",
            title=f"Absolute frequency difference: {col}, {df_name}",
        )

        fig.update_layout(
            xaxis_title=col,
            yaxis_title="|Δ proportion|",
            template="plotly_white",
            height=400,
        )

        fig.show()
        fig.write_image(f"figures/abs_diff_{col}_{df_name}.png")


def plot_marginal_relative_diff_plotly(df, sampled_df, cols, df_name=""):
    """Plot relative difference in normalized value counts using Plotly."""
    import plotly.express as px
    import pandas as pd
    import numpy as np
    import kaleido

    kaleido.start_sync_server(n=5)

    for col in cols:

        # Normalized value counts
        full = df[col].value_counts(normalize=True)
        samp = sampled_df[col].value_counts(normalize=True)

        # Align indexes to avoid NaNs for missing categories
        combined_index = full.index.union(samp.index)
        full = full.reindex(combined_index, fill_value=0)
        samp = samp.reindex(combined_index, fill_value=0)

        # Relative difference (safe division)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = (full - samp) / samp
            rel_diff = rel_diff.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Sort by absolute relative difference (largest deviations first)
        rel_diff = rel_diff.reindex(rel_diff.abs().sort_values(ascending=False).index)

        # Convert to DataFrame for Plotly
        rel_diff_df = rel_diff.reset_index()
        rel_diff_df.columns = [col, "Relative Δ proportion"]

        fig = px.bar(
            rel_diff_df,
            x=col,
            y="Relative Δ proportion",
            title=f"Relative frequency difference: {col}, {df_name}",
        )

        fig.update_layout(
            xaxis_title=col,
            yaxis_title="Relative Δ proportion",
            template="plotly_white",
            height=400,
        )

        fig.show()
        fig.write_image(f"figures/rel_diff_{col}_{df_name}.png")


def plot_value_counts(data, columns):
    """Plot value counts for each column in descending order using Plotly."""
    import plotly.express as px
    import kaleido

    kaleido.start_sync_server(n=5)

    for col in columns:
        # Get value counts sorted descending
        value_counts = data[col].value_counts().reset_index()
        value_counts.columns = [col, "Count"]

        # Reverse for horizontal bar chart (largest at top)
        value_counts = value_counts.sort_values("Count", ascending=True)

        fig = px.bar(
            value_counts,
            x="Count",
            y=col,
            orientation="h",
            title=f"Value counts for {col}",
        )

        fig.update_layout(
            yaxis_title=col, xaxis_title="Count", template="plotly_white", height=400
        )

        fig.show()
        fig.write_image(f"figures/value_counts_{col}.png")
