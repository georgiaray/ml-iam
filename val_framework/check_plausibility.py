"""
Plausibility check for ML-IAM XGBoost predictions (Shin et al. strategy).

For each predicted trajectory (scenario x region x variable), computes the
5-year period-on-period growth rate:

    g_y = (x_y - x_{y-5}) / x_{y-5}

Derives empirical bounds from the AR6 test-set ground truth, then flags any
predicted timestep where the growth rate falls outside those bounds.

Usage:
    python scripts/check_plausibility.py --run_id xgb_02
    python scripts/check_plausibility.py --run_id xgb_02 --percentile 5
    python scripts/check_plausibility.py --run_id xgb_02 --by_category
"""

import argparse
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

class _Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, path: Path):
        self._file = open(path, "w")
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()


# Ensure repo root is on the path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_run_artifacts(run_id: str):
    """Load test_data, y_test (scaled), preds (scaled), y_scaler for a run."""
    from src.utils.run_store import RunStore
    from scripts.train_xgb import derive_splits

    print(f"\n{'='*60}")
    print(f"Loading artifacts for run: {run_id}")
    print(f"{'='*60}")

    store = RunStore(run_id)

    print("  Loading cached processed data (parquet)...")
    data = store.load_processed_data()
    print(f"  Processed data shape: {data.shape}")

    print("  Deriving train/val/test splits...")
    splits = derive_splits(data)

    test_data = splits["test_data"]
    y_test_scaled = splits["y_test"]
    targets = splits["targets"]
    y_scaler = splits["y_scaler"]

    print(f"  Test set rows: {len(test_data)}")
    print(f"  Target variables ({len(targets)}): {targets}")

    print("  Loading predictions from artifacts/predictions.pkl...")
    pred_bundle = store.load_predictions()
    preds_scaled = pred_bundle["preds"]
    print(f"  Predictions array shape: {preds_scaled.shape}")

    return test_data, y_test_scaled, preds_scaled, y_scaler, targets


def unscale(arr, scaler):
    """Inverse-transform a scaled numpy array back to original units."""
    return scaler.inverse_transform(arr)


def build_trajectory_df(test_data: pd.DataFrame, values: np.ndarray, targets: list, label: str) -> pd.DataFrame:
    """
    Combine test_data index columns with a (n_rows, n_targets) values array
    into a long-format DataFrame: [Model, Scenario, Region, Scenario_Category, Year, Variable, Value].
    """
    index_cols = ["Model", "Scenario", "Region", "Scenario_Category", "Year"]
    idx = test_data[index_cols].reset_index(drop=True)

    wide = pd.DataFrame(values, columns=targets)
    combined = pd.concat([idx, wide], axis=1)

    long = combined.melt(
        id_vars=index_cols,
        value_vars=targets,
        var_name="Variable",
        value_name="Value",
    )
    long["source"] = label
    return long.reset_index(drop=True)


def compute_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a long-format DataFrame with columns
    [Model, Scenario, Region, Year, Variable, Value],
    compute 5-year period growth rate for each trajectory.

    g_y = (x_y - x_{y-5}) / x_{y-5}

    Rows where x_{y-5} == 0 (or NaN) are dropped to avoid div-by-zero.
    """
    group_cols = ["Model", "Scenario", "Region", "Variable"]
    df = df.sort_values(group_cols + ["Year"]).copy()
    df["Value_lag5"] = df.groupby(group_cols)["Value"].shift(1)
    df["growth_rate"] = (df["Value"] - df["Value_lag5"]) / df["Value_lag5"].abs()

    before = len(df)
    df = df.dropna(subset=["growth_rate"])
    df = df[np.isfinite(df["growth_rate"])]
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with NaN/inf growth rates (zero denominators or missing lags)")

    return df


def derive_empirical_bounds(gt_growth: pd.DataFrame, lower_pct: float, upper_pct: float) -> pd.DataFrame:
    """
    For each Variable, compute (lower_pct, upper_pct) percentile bounds
    from the ground-truth growth rate distribution.
    """
    stats = gt_growth.groupby("Variable")["growth_rate"].describe(
        percentiles=[lower_pct / 100, upper_pct / 100]
    ).reset_index()
    # pandas formats percentile columns as "1%" not "1.0%"
    lower_col = next(c for c in stats.columns if c.endswith("%") and float(c[:-1]) == lower_pct)
    upper_col = next(c for c in stats.columns if c.endswith("%") and float(c[:-1]) == upper_pct)
    bounds = stats[["Variable", lower_col, upper_col]].rename(
        columns={lower_col: "lower_bound", upper_col: "upper_bound"}
    )
    return bounds


def flag_violations(pred_growth: pd.DataFrame, bounds: pd.DataFrame) -> pd.DataFrame:
    """
    Join predicted growth rates with bounds and flag any out-of-range timesteps.
    Returns the pred_growth DataFrame with added columns:
        lower_bound, upper_bound, violation, severity
    """
    merged = pred_growth.merge(bounds, on="Variable", how="left")
    merged["violation"] = (
        (merged["growth_rate"] < merged["lower_bound"]) |
        (merged["growth_rate"] > merged["upper_bound"])
    )
    # Severity: how many bound-widths outside the range
    bound_width = (merged["upper_bound"] - merged["lower_bound"]).abs()
    overshoot = np.maximum(
        merged["lower_bound"] - merged["growth_rate"],
        merged["growth_rate"] - merged["upper_bound"],
    ).clip(lower=0)
    merged["severity"] = overshoot / bound_width.replace(0, np.nan)
    return merged


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_overall(flagged: pd.DataFrame, bounds: pd.DataFrame, lower_pct: float, upper_pct: float):
    total = len(flagged)
    n_viol = flagged["violation"].sum()
    rate = 100 * n_viol / total if total > 0 else 0.0

    print(f"\n{'='*60}")
    print("OVERALL VIOLATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Bounds used          : {lower_pct}th / {upper_pct}th percentile of AR6 ground truth")
    print(f"  Total predicted timesteps evaluated : {total:,}")
    print(f"  Violations           : {n_viol:,}  ({rate:.2f}%)")
    print(f"  Mean severity (of violations): "
          f"{flagged.loc[flagged['violation'], 'severity'].mean():.3f} bound-widths")


def report_by_variable(flagged: pd.DataFrame, bounds: pd.DataFrame):
    print(f"\n{'='*60}")
    print("VIOLATIONS BY VARIABLE")
    print(f"{'='*60}")
    summary = (
        flagged.groupby("Variable")
        .agg(
            total=("violation", "count"),
            violations=("violation", "sum"),
            mean_severity=("severity", lambda x: x[flagged.loc[x.index, "violation"]].mean()),
        )
        .reset_index()
    )
    summary["rate_%"] = 100 * summary["violations"] / summary["total"]
    summary = summary.sort_values("rate_%", ascending=False)

    for _, row in summary.iterrows():
        b = bounds[bounds["Variable"] == row["Variable"]].iloc[0]
        print(
            f"  {row['Variable']:<35}  "
            f"violations: {int(row['violations']):>5} / {int(row['total']):>6}  "
            f"({row['rate_%']:5.1f}%)  "
            f"bounds: [{b['lower_bound']:+.3f}, {b['upper_bound']:+.3f}]  "
            f"mean severity: {row['mean_severity']:.3f}"
        )


def report_by_category(flagged: pd.DataFrame):
    print(f"\n{'='*60}")
    print("VIOLATIONS BY SCENARIO CATEGORY")
    print(f"{'='*60}")
    summary = (
        flagged.groupby("Scenario_Category")
        .agg(total=("violation", "count"), violations=("violation", "sum"))
        .reset_index()
    )
    summary["rate_%"] = 100 * summary["violations"] / summary["total"]
    summary = summary.sort_values("rate_%", ascending=False)

    for _, row in summary.iterrows():
        print(
            f"  {str(row['Scenario_Category']):<20}  "
            f"violations: {int(row['violations']):>5} / {int(row['total']):>6}  "
            f"({row['rate_%']:5.1f}%)"
        )


def report_top_violators(flagged: pd.DataFrame, n: int = 10):
    print(f"\n{'='*60}")
    print(f"TOP {n} MOST VIOLATED SCENARIOS (by violation count)")
    print(f"{'='*60}")
    viol = flagged[flagged["violation"]]
    top = (
        viol.groupby(["Model", "Scenario", "Region"])
        .agg(violations=("violation", "sum"), mean_severity=("severity", "mean"))
        .reset_index()
        .sort_values("violations", ascending=False)
        .head(n)
    )
    for _, row in top.iterrows():
        print(
            f"  {row['Model']:<25} | {row['Scenario']:<30} | {row['Region']:<20}  "
            f"violations: {int(row['violations'])}  mean severity: {row['mean_severity']:.3f}"
        )


def report_gt_distribution(gt_growth: pd.DataFrame):
    print(f"\n{'='*60}")
    print("GROUND TRUTH GROWTH RATE DISTRIBUTION (per variable)")
    print(f"{'='*60}")
    stats = (
        gt_growth.groupby("Variable")["growth_rate"]
        .describe(percentiles=[0.01, 0.05, 0.25, 0.75, 0.95, 0.99])
        .reset_index()
    )
    for _, row in stats.iterrows():
        print(
            f"  {row['Variable']:<35}  "
            f"n={int(row['count']):>6}  "
            f"p1={row['1%']:+.3f}  p5={row['5%']:+.3f}  "
            f"p25={row['25%']:+.3f}  p75={row['75%']:+.3f}  "
            f"p95={row['95%']:+.3f}  p99={row['99%']:+.3f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plausibility check for ML-IAM predictions")
    parser.add_argument("--run_id", required=True, help="Run ID, e.g. xgb_02")
    parser.add_argument(
        "--percentile", type=float, default=1.0,
        help="Lower tail percentile for bounds (default: 1). Upper = 100 - percentile."
    )
    parser.add_argument(
        "--by_category", action="store_true",
        help="Also break down violations by scenario category (C1-C8)"
    )
    parser.add_argument(
        "--use_ground_truth", action="store_true",
        help="Check AR6 ground truth (y_test) instead of model predictions. "
             "Bounds are still derived from y_test; this checks whether the "
             "ground truth itself falls within those bounds."
    )
    args = parser.parse_args()

    lower_pct = args.percentile
    upper_pct = 100.0 - args.percentile

    # ---- Load ----
    test_data, y_test_scaled, preds_scaled, y_scaler, targets = load_run_artifacts(args.run_id)

    # ---- Unscale to original units ----
    print(f"\n  Inverse-transforming predictions and ground truth to original units...")
    y_test = unscale(y_test_scaled, y_scaler)
    preds = unscale(preds_scaled, y_scaler)
    print(f"  Ground truth value range (all targets): [{y_test.min():.3f}, {y_test.max():.3f}]")
    print(f"  Predictions value range  (all targets): [{preds.min():.3f}, {preds.max():.3f}]")

    if args.use_ground_truth:
        values     = y_test
        label      = "AR6 ground truth (y_test)"
        out_subdir = "plausibility_ground_truth"
    else:
        values     = preds
        label      = "model predictions"
        out_subdir = "plausibility"

    print(f"\n  Data source: {label}")

    # ---- Build trajectory DataFrames ----
    print(f"\n  Building long-format trajectory DataFrames...")
    gt_long   = build_trajectory_df(test_data, y_test,  targets, label="ground_truth")
    val_long  = build_trajectory_df(test_data, values,  targets, label=label)
    print(f"  Ground truth long shape : {gt_long.shape}")
    print(f"  Validated data shape    : {val_long.shape}")
    n_scenarios = gt_long[["Model", "Scenario", "Region"]].drop_duplicates().shape[0]
    print(f"  Unique scenario x region combinations: {n_scenarios}")

    # ---- Compute growth rates ----
    print(f"\n  Computing 5-year period growth rates...")
    gt_growth  = compute_growth_rates(gt_long)
    val_growth = compute_growth_rates(val_long)
    print(f"  Ground truth growth rate rows : {len(gt_growth):,}")
    print(f"  Validated data growth rate rows: {len(val_growth):,}")

    # ---- Empirical bounds always derived from ground truth ----
    print(f"\n  Deriving empirical bounds from ground truth ({lower_pct}th / {upper_pct}th percentiles)...")
    bounds = derive_empirical_bounds(gt_growth, lower_pct, upper_pct)
    print(f"\n  {'Variable':<35}  {'Lower':>8}  {'Upper':>8}")
    print(f"  {'-'*55}")
    for _, row in bounds.iterrows():
        print(f"  {row['Variable']:<35}  {row['lower_bound']:>+8.4f}  {row['upper_bound']:>+8.4f}")

    # ---- Flag violations in the validated data ----
    print(f"\n  Flagging violations in {label}...")
    flagged = flag_violations(val_growth, bounds)

    # ---- Tee stdout to report.txt from here onwards ----
    out_dir = REPO_ROOT / "results" / "xgb" / args.run_id / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.txt"
    sys.stdout = _Tee(report_path)

    # ---- Reports ----
    report_gt_distribution(gt_growth)
    report_overall(flagged, bounds, lower_pct, upper_pct)
    report_by_variable(flagged, bounds)

    if args.by_category:
        report_by_category(flagged)

    report_top_violators(flagged)

    # ---- Save results ----
    out_path = out_dir / "growth_rate_violations.csv"
    flagged.to_csv(out_path, index=False)
    bounds_path = out_dir / "empirical_bounds.csv"
    bounds.to_csv(bounds_path, index=False)
    report_path = out_dir / "report.txt"
    print(f"\n{'='*60}")
    print(f"Results saved to:")
    print(f"  {out_path}")
    print(f"  {bounds_path}")
    print(f"  {report_path}")
    print(f"{'='*60}\n")

    # Flush and close tee so report.txt is fully written
    if isinstance(sys.stdout, _Tee):
        sys.stdout.close()


if __name__ == "__main__":
    main()