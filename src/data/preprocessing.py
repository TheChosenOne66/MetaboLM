"""Missing-value imputation and feature normalization for NMR metabolomics.

All transformations are *fit* on the training split and *applied* to both
training and validation to prevent data leakage.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Imputation
# ──────────────────────────────────────────────────────────────────────────────

def impute_missing_simple(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median imputation: fit on *train_df*, transform both splits.

    Parameters
    ----------
    train_df, val_df:
        DataFrames that must contain all columns listed in *feature_cols*.
    feature_cols:
        Column names to impute (typically the 168 metabolite columns).

    Returns
    -------
    ``(train_imputed, val_imputed)`` — copies with NaNs replaced by the
    training-set column medians.
    """
    # Compute medians on train only
    medians = train_df[feature_cols].median()

    train_out = train_df.copy()
    val_out = val_df.copy()

    n_miss_train = int(train_out[feature_cols].isna().sum().sum())
    n_miss_val = int(val_out[feature_cols].isna().sum().sum())

    train_out[feature_cols] = train_out[feature_cols].fillna(medians)
    val_out[feature_cols] = val_out[feature_cols].fillna(medians)

    logger.info(
        "Median imputation: filled %d train cells, %d val cells across %d features.",
        n_miss_train, n_miss_val, len(feature_cols),
    )
    return train_out, val_out


def impute_missing_mice(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    num_datasets: int = 3,
    num_iterations: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MICE imputation using *miceforest* (optional, slower).

    Falls back to :func:`impute_missing_simple` if miceforest is not installed
    or if the data is too large (> 200 K rows) for practical runtime.

    Parameters
    ----------
    train_df, val_df:
        Input DataFrames.
    feature_cols:
        Columns to impute.
    num_datasets, num_iterations:
        miceforest parameters.
    seed:
        Random seed.

    Returns
    -------
    ``(train_imputed, val_imputed)``.
    """
    MAX_ROWS_FOR_MICE = 200_000

    if len(train_df) > MAX_ROWS_FOR_MICE:
        logger.warning(
            "Training set has %d rows (> %d), falling back to median imputation.",
            len(train_df), MAX_ROWS_FOR_MICE,
        )
        return impute_missing_simple(train_df, val_df, feature_cols)

    try:
        import miceforest as mf
    except ImportError:
        logger.warning("miceforest not installed, falling back to median imputation.")
        return impute_missing_simple(train_df, val_df, feature_cols)

    train_out = train_df.copy()
    val_out = val_df.copy()

    # Create kernel on training data
    kernel = mf.ImputationKernel(
        train_out[feature_cols],
        datasets=num_datasets,
        save_all_iterations=False,
        random_state=seed,
    )
    kernel.mice(num_iterations)

    # Average across multiple imputations
    imputed_arrays = [kernel.complete_data(dataset=d) for d in range(num_datasets)]
    train_imputed = sum(imputed_arrays) / num_datasets  # type: ignore[arg-type]
    train_out[feature_cols] = train_imputed.values

    # Impute validation using the fitted kernel
    val_imputed = kernel.impute_new_data(val_out[feature_cols])
    val_avg = sum(val_imputed.complete_data(dataset=d) for d in range(num_datasets)) / num_datasets  # type: ignore[arg-type]
    val_out[feature_cols] = val_avg.values

    logger.info("MICE imputation complete (%d datasets, %d iterations).", num_datasets, num_iterations)
    return train_out, val_out


# ──────────────────────────────────────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────────────────────────────────────

def zscore_normalize(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Z-score normalization: fit on *train_df*, transform both.

    Parameters
    ----------
    train_df, val_df:
        DataFrames containing *feature_cols*.
    feature_cols:
        Column names to normalize.

    Returns
    -------
    ``(train_normed, val_normed, means, stds)`` where *means* and *stds* are
    ``pd.Series`` indexed by *feature_cols*, fitted on training data only.
    Columns with zero standard deviation are left unscaled (std replaced by 1).
    """
    means = train_df[feature_cols].mean()
    stds = train_df[feature_cols].std()

    # Avoid division by zero for constant features
    zero_std = stds == 0
    if zero_std.any():
        n_zero = int(zero_std.sum())
        logger.warning("%d features have zero std in training set — left unscaled.", n_zero)
        stds = stds.replace(0, 1)

    train_out = train_df.copy()
    val_out = val_df.copy()

    train_out[feature_cols] = (train_out[feature_cols] - means) / stds
    val_out[feature_cols] = (val_out[feature_cols] - means) / stds

    logger.info("Z-score normalization applied to %d features.", len(feature_cols))
    return train_out, val_out, means, stds


def rank_inverse_normal(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank-based inverse-normal transformation (alternative to z-score).

    Each feature is replaced by its rank (within the split), scaled to (0, 1),
    and passed through the inverse-normal (probit) function.  Fitted
    independently on each split (rank transformation is non-parametric).

    Returns
    -------
    ``(train_transformed, val_transformed)``.
    """
    from scipy.stats import norm

    def _rint(series: pd.Series) -> pd.Series:
        """Rank inverse-normal for a single column."""
        ranked = series.rank(method="average")
        # Blom offset: (r - 3/8) / (n + 1/4)
        n = series.notna().sum()
        uniform = (ranked - 0.375) / (n + 0.25)
        # Clip to (eps, 1-eps) to avoid infinities
        eps = 1e-6
        uniform = uniform.clip(eps, 1 - eps)
        return pd.Series(norm.ppf(uniform), index=series.index)

    train_out = train_df.copy()
    val_out = val_df.copy()

    for col in feature_cols:
        train_out[col] = _rint(train_out[col])
        val_out[col] = _rint(val_out[col])

    logger.info("Rank-inverse-normal transformation applied to %d features.", len(feature_cols))
    return train_out, val_out


# ──────────────────────────────────────────────────────────────────────────────
# Outlier handling
# ──────────────────────────────────────────────────────────────────────────────

def winsorize(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    lower_quantile: float = 0.001,
    upper_quantile: float = 0.999,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Winsorize extreme values: clip to training-set quantiles.

    Parameters
    ----------
    lower_quantile, upper_quantile:
        Quantile thresholds computed on *train_df*.

    Returns
    -------
    ``(train_clipped, val_clipped)``.
    """
    lows = train_df[feature_cols].quantile(lower_quantile)
    highs = train_df[feature_cols].quantile(upper_quantile)

    train_out = train_df.copy()
    val_out = val_df.copy()

    train_out[feature_cols] = train_out[feature_cols].clip(lower=lows, upper=highs, axis=1)
    val_out[feature_cols] = val_out[feature_cols].clip(lower=lows, upper=highs, axis=1)

    logger.info(
        "Winsorized to [%.4f, %.4f] quantiles on %d features.",
        lower_quantile, upper_quantile, len(feature_cols),
    )
    return train_out, val_out
