import pandas as pd
import numpy as np
from itables import init_notebook_mode

init_notebook_mode(all_interactive=True)


def balanced_weighted_sample(df, balance_cols, N, power=1.0, random_state=0):
    weights = pd.Series(1.0, index=df.index)

    for col in balance_cols:
        freq = df[col].value_counts()
        col_weight = df[col].map(
            lambda x: (
                1 / (freq[x] * 8)
                if x == "immune"
                else (
                    1 / (freq[x] * 2)
                    if (x == "AFOG/Masson Trichrom" or x == "HE/HES")
                    else 1 / freq[x]
                )
            )
        )
        weights *= col_weight**power
    has_crescent = df["number_glom_crescent"] > 0.5
    weights = weights + weights * 3 * has_crescent.astype(float)
    weights /= weights.sum()

    return df.sample(n=N, weights=weights, random_state=random_state)
