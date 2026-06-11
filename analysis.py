"""EDA, feature-selection and evaluation statistics.

Everything here returns DataFrames / Series / dicts and plots nothing. The
dashboard turns these into figures; `python analysis.py` prints them as text.
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from hourly_dataset import (
    FEATURE_COLUMNS, TEMP_COLUMNS, TARGET_COLUMN,
    build_hourly_dataset, split_features_target,
)
from hourly_model import build_model, run_holdout
from price_cache import load_cached_prices


# ---- 1. exploratory data analysis ----

def price_summary(df: pd.DataFrame) -> pd.Series:
    """Summary statistics for the target (tomorrow's hourly price)."""
    p = df[TARGET_COLUMN]
    return pd.Series({
        "count": len(p),
        "mean": p.mean(),
        "std": p.std(),
        "min": p.min(),
        "25%": p.quantile(0.25),
        "median": p.median(),
        "75%": p.quantile(0.75),
        "max": p.max(),
        "% negative hours": (p < 0).mean() * 100,
        "intraday range (median)": (
            df.groupby("date")[TARGET_COLUMN].agg(lambda s: s.max() - s.min()).median()
        ),
    })


def hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / std / quartiles of price by local hour (the duck curve)."""
    return (
        df.groupby("hour")[TARGET_COLUMN]
        .agg(mean="mean", std="std",
             q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
        .reset_index()
    )


def binned_price(df: pd.DataFrame, x_col: str, bins: int = 24) -> pd.DataFrame:
    """
    Mean target price within equal-width bins of `x_col` (wind, temp, ...),
    as center / mean / count rows. Shows how price responds to a driver.
    """
    cut = pd.cut(df[x_col], bins=bins)
    grouped = df.groupby(cut, observed=True)[TARGET_COLUMN].agg(["mean", "count"])
    grouped["center"] = [iv.mid for iv in grouped.index]
    return grouped.reset_index(drop=True)


def resolution_breakdown(zone: str = "SE4") -> pd.Series:
    """
    Records per day in the raw price cache: 24 = hourly, 96 = 15-minute
    (2025+), 23/25 and 92/100 = DST clock-change days. This is why the
    loader aggregates everything to hourly.
    """
    prices = load_cached_prices(zone)
    local_date = prices["timestamp_utc"].dt.tz_convert("Europe/Stockholm").dt.date
    per_day = prices.groupby(local_date).size()
    return per_day.value_counts().sort_index()


# ---- 2. feature selection ----

def feature_target_correlations(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Pearson r between each feature and the target, sorted by |r|."""
    r = {col: stats.pearsonr(X[col], y)[0] for col in X.columns}
    return pd.Series(r).reindex(pd.Series(r).abs().sort_values(ascending=False).index)


def correlation_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix among features (spot redundant pairs)."""
    return X.corr()


def anova_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-way ANOVA of the price across each categorical split. Reports F and
    p plus eta-squared, the share of price variance the split explains. With
    ~40k rows p is ~0 for everything, so eta^2 is what actually ranks them.
    """
    grand_mean = df[TARGET_COLUMN].mean()
    ss_total = ((df[TARGET_COLUMN] - grand_mean) ** 2).sum()

    rows = []
    for col in ["hour", "day_of_week_tomorrow", "is_weekend_tomorrow", "is_holiday_tomorrow"]:
        groups = [g[TARGET_COLUMN].to_numpy() for _, g in df.groupby(col)]
        f, p = stats.f_oneway(*groups)
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        rows.append({"feature": col, "F_statistic": f, "p_value": p,
                     "variance_explained": ss_between / ss_total})
    return pd.DataFrame(rows).sort_values("variance_explained", ascending=False)


# Ablation groups: the raw columns behind each real-world driver.
# (No temperature group - those columns are out of the model, see README.)
FEATURE_GROUPS = {
    "price anchors": ["price_today_same_hour", "price_today_daily_mean"],
    "wind": ["wind_today_mean", "wind_tomorrow_hour", "wind_delta_hour"],
    "hour": ["hour_sin", "hour_cos"],
    "calendar": ["day_of_week_tomorrow", "is_weekend_tomorrow", "is_holiday_tomorrow"],
    "season": ["day_of_year_sin", "day_of_year_cos", "month"],
}


def ablation_study(df: pd.DataFrame, holdout_months: int = 4) -> pd.DataFrame:
    """
    Drop-column ablation: retrain without each feature group, measure the
    rise in holdout MAE. Catches the non-linear and interaction effects that
    Pearson r misses, which is the point for a tree model. Returns group,
    max_abs_r (strongest linear correlation in the group), mae, delta_mae
    (positive = the group helps).
    """
    cutoff = df["date"].max() - pd.DateOffset(months=holdout_months)
    train, test = df[df["date"] <= cutoff], df[df["date"] > cutoff]
    y_train, y_test = train[TARGET_COLUMN], test[TARGET_COLUMN]

    def mae(cols):
        model = build_model().fit(train[cols], y_train)
        return mean_absolute_error(y_test, model.predict(test[cols]))

    full = mae(FEATURE_COLUMNS)
    abs_r = feature_target_correlations(train[FEATURE_COLUMNS], y_train).abs()

    rows = [{"group": "(full model)", "max_abs_r": np.nan, "mae": full, "delta_mae": 0.0}]
    for group, cols in FEATURE_GROUPS.items():
        kept = [c for c in FEATURE_COLUMNS if c not in cols]
        d = mae(kept)
        rows.append({"group": group, "max_abs_r": abs_r[cols].max(),
                     "mae": d, "delta_mae": d - full})
    return pd.DataFrame(rows)


def seasonal_backtest(
    df: pd.DataFrame,
    variants: dict[str, list[str]] | None = None,
    n_folds: int = 6,
    fold_months: int = 4,
    seeds: tuple[int, ...] = (42, 0, 7),
) -> pd.DataFrame:
    """
    MAE per chronological fold for competing feature sets, stepping back
    `fold_months` at a time with an expanding training window. One holdout
    can mislead seasonally (the standard one is mostly spring), and a single
    fit carries seed noise because early stopping splits off a random
    validation set - hence folds x seeds.

    Default variants re-test the temperature decision: current features vs
    the same plus the temp_* columns. Pass any {name: columns} dict to test
    something else; the columns must exist in `df`. Slow: fits
    len(variants) x n_folds x len(seeds) models.
    """
    if variants is None:
        variants = {"model": FEATURE_COLUMNS,
                    "with temperature": FEATURE_COLUMNS + TEMP_COLUMNS}

    hi = df["date"].max()
    rows = []
    for _ in range(n_folds):
        lo = hi - pd.DateOffset(months=fold_months)
        train = df[df["date"] <= lo]
        test = df[(df["date"] > lo) & (df["date"] <= hi)]
        if len(train) < 5000 or test.empty:
            break  # ran out of history
        y_tr, y_te = train[TARGET_COLUMN], test[TARGET_COLUMN]
        row = {
            "test window": f"{(lo + pd.Timedelta(days=1)).date()} .. {hi.date()}",
            "n": len(test),
            "baseline": mean_absolute_error(y_te, test["price_today_same_hour"]),
        }
        for name, cols in variants.items():
            row[name] = float(np.mean([
                mean_absolute_error(
                    y_te,
                    build_model(random_state=s).fit(train[cols], y_tr).predict(test[cols]),
                )
                for s in seeds
            ]))
        rows.append(row)
        hi = lo
    return pd.DataFrame(rows)


# ---- 3. evaluation ----

def evaluation_metrics(actual, predicted) -> dict:
    """MAE, RMSE, R² and mean bias (SEK/kWh)."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": root_mean_squared_error(actual, predicted),
        "R2": r2_score(actual, predicted),
        "bias": float(np.mean(predicted - actual)),
    }


def per_hour_mae(results: pd.DataFrame) -> pd.DataFrame:
    """Model vs baseline MAE for each local hour."""
    err = results.assign(
        model=(results["predicted"] - results["actual"]).abs(),
        base=(results["baseline"] - results["actual"]).abs(),
    )
    return err.groupby("hour")[["model", "base"]].mean().reset_index()


def margin_vs_baseline(results: pd.DataFrame) -> pd.Series:
    """
    Per-hour margin over the baseline: |baseline error| - |model error|
    (positive = model was closer that hour). Its mean is baseline MAE minus
    model MAE. Worth having next to the win rate: winning often-by-a-little
    while losing rarely-but-big would show up here, not there.
    """
    return ((results["baseline"] - results["actual"]).abs()
            - (results["predicted"] - results["actual"]).abs())


def permutation_importances(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5) -> pd.Series:
    """Permutation importance (mean rise in MAE when a feature is shuffled)."""
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats,
        scoring="neg_mean_absolute_error", random_state=42,
    )
    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)


# ---- CLI: print a text summary of everything ----

def _main():
    parser = argparse.ArgumentParser(description="Text summary of the analysis statistics")
    parser.add_argument("--backtest", action="store_true",
                        help="Also run the seasonal feature backtest (slow, ~36 model fits).")
    args = parser.parse_args()

    df = build_hourly_dataset(verbose=True)
    print("\n== Price summary ==")
    print(price_summary(df).round(3).to_string())

    print("\n== Records per day (data quality) ==")
    print(resolution_breakdown().to_string())

    X, y = split_features_target(df)
    print("\n== |Pearson r| with target (top) ==")
    print(feature_target_correlations(X, y).round(3).head(8).to_string())

    print("\n== ANOVA (price across groups) ==")
    print(anova_by_group(df).round(4).to_string(index=False))

    print("\n== Ablation: holdout MAE when each group is dropped ==")
    print("   (low |r| can still matter a lot - see 'hour')")
    print(ablation_study(df).round(4).to_string(index=False))

    print("\n== Holdout metrics ==")
    h = run_holdout(df=df, verbose=False)
    print("  model   :", {k: round(v, 4) for k, v in evaluation_metrics(h["results"]["actual"], h["results"]["predicted"]).items()})
    print("  baseline:", {k: round(v, 4) for k, v in evaluation_metrics(h["results"]["actual"], h["results"]["baseline"]).items()})
    m = margin_vs_baseline(h["results"])
    print(f"  margin  : {m.mean():+.4f} SEK/kWh per hour on average; "
          f"ahead on {(m > 0).mean() * 100:.0f}% of hours "
          f"(avg win {m[m > 0].mean():+.3f}, avg loss {m[m <= 0].mean():+.3f})")

    if args.backtest:
        print("\n== Seasonal backtest: MAE per fold, with vs without temperature ==")
        bt = seasonal_backtest(df)
        print(bt.round(4).to_string(index=False))
        print("\n  mean over folds:")
        print(bt.drop(columns=["test window", "n"]).mean().round(4).to_string())


if __name__ == "__main__":
    _main()
