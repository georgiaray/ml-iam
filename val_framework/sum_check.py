"""
Sum check for ML-IAM XGBoost predictions.

Verifies that Secondary Energy|Electricity (total) matches the sum of its
sub-fuel components at every timestep, for every scenario x region combination.

Error metric (per timestep):
    e = |total - sum_of_components| / |total|

A scenario passes if e < 1.2% at ALL timesteps.

Usage:
    python scripts/sum_check.py --run_id xgb_03
    python scripts/sum_check.py --run_id xgb_03 --threshold 0.05
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TOTAL_VAR = "Secondary Energy|Electricity"
COMPONENT_VARS = [
    "Secondary Energy|Electricity|Biomass",
    "Secondary Energy|Electricity|Coal",
    "Secondary Energy|Electricity|Gas",
    "Secondary Energy|Electricity|Geothermal",
    "Secondary Energy|Electricity|Hydro",
    "Secondary Energy|Electricity|Nuclear",
    "Secondary Energy|Electricity|Oil",
    "Secondary Energy|Electricity|Solar",
    "Secondary Energy|Electricity|Wind",
]


# ---------------------------------------------------------------------------
# Tee (mirror stdout to file)
# ---------------------------------------------------------------------------

class _Tee:
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


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_predictions(run_id: str):
    """
    Returns a long-format DataFrame with columns:
        Model, Scenario, Region, Scenario_Category, Year, Variable, Value
    containing only the variables needed for the sum check.
    """
    from src.utils.run_store import RunStore
    from scripts.train_xgb import derive_splits

    print(f"\n{'='*60}")
    print(f"Loading artifacts for run: {run_id}")
    print(f"{'='*60}")

    store = RunStore(run_id)

    print("  Loading cached processed data...")
    data = store.load_processed_data()
    print(f"  Processed data shape: {data.shape}")

    print("  Deriving splits...")
    splits = derive_splits(data)

    test_data = splits["test_data"]
    y_test_scaled = splits["y_test"]
    targets = splits["targets"]
    y_scaler = splits["y_scaler"]

    print(f"  Targets ({len(targets)}): {targets}")

    # Check required variables are present
    required = [TOTAL_VAR] + COMPONENT_VARS
    missing = [v for v in required if v not in targets]
    if missing:
        print(f"\nERROR: The following required variables are not in this run's targets:")
        for v in missing:
            print(f"  - {v}")
        print("\nThis check requires the model to have been trained with all Secondary")
        print("Energy|Electricity variables as outputs. Re-train with the full variable set.")
        sys.exit(1)

    print("  Loading predictions...")
    pred_bundle = store.load_predictions()
    preds_scaled = pred_bundle["preds"]

    print("  Inverse-transforming to original units...")
    preds = y_scaler.inverse_transform(preds_scaled)
    y_test = y_scaler.inverse_transform(y_test_scaled)
    print(f"  Predictions shape: {preds.shape}")

    return test_data, preds, y_test, targets, required


def build_long(test_data, values, targets, required, label):
    """Build long-format DataFrame from a values array."""
    index_cols = ["Model", "Scenario", "Region", "Scenario_Category", "Year"]
    idx = test_data[index_cols].reset_index(drop=True)
    wide = pd.DataFrame(values, columns=targets)
    combined = pd.concat([idx, wide], axis=1)
    long = combined.melt(
        id_vars=index_cols,
        value_vars=required,
        var_name="Variable",
        value_name="Value",
    )
    print(f"  Long-format rows ({label}, relevant variables only): {len(long):,}")
    return long


# ---------------------------------------------------------------------------
# Sum check
# ---------------------------------------------------------------------------

def run_sum_check(long: pd.DataFrame, threshold: float, abs_floor: float = 1.0) -> pd.DataFrame:
    """
    For each (Model, Scenario, Region, Year), compute:
        sum_components = sum of COMPONENT_VARS
        total          = TOTAL_VAR
        error          = |total - sum_components| / |total|

    Returns a DataFrame indexed by (Model, Scenario, Region, Year) with
    columns: total, sum_components, zero_total, abs_error, passed_timestep.

    Timesteps where |total| < abs_floor are flagged as zero_total=True and
    abs_error is set to NaN (relative error is unreliable for near-zero
    denominators). These count as failed timesteps in passed_timestep (False)
    and are visible via n_zero_total in scenario_summary.
    """
    # Check for duplicates before pivoting — silently taking "first" would
    # mask upstream data issues.
    index_cols = ["Model", "Scenario", "Region", "Scenario_Category", "Year", "Variable"]
    dupes = long[long.duplicated(subset=index_cols)]
    if not dupes.empty:
        raise ValueError(
            f"{len(dupes)} duplicate rows found in input data "
            f"(e.g. {dupes[index_cols].iloc[0].to_dict()}). "
            "Check your splits/data loading pipeline."
        )

    pivot = long.pivot_table(
        index=["Model", "Scenario", "Region", "Scenario_Category", "Year"],
        columns="Variable",
        values="Value",
        aggfunc="first",
    ).reset_index()

    pivot["sum_components"] = pivot[COMPONENT_VARS].sum(axis=1)
    pivot["total"] = pivot[TOTAL_VAR]

    # Exclude near-zero totals: relative error is unreliable when the denominator
    # is too small. abs_floor defaults to 1.0 (in the data's native units).
    # In practice this is a safety net — real data should be well above this.
    pivot["zero_total"] = pivot["total"].abs() < abs_floor
    pivot["abs_error"] = np.where(
        pivot["zero_total"],
        np.nan,
        (pivot["total"] - pivot["sum_components"]).abs() / pivot["total"].abs(),
    )
    # NaN < threshold evaluates to False, so zero_total rows count as failed.
    pivot["passed_timestep"] = pivot["abs_error"] < threshold

    return pivot


def scenario_summary(timestep_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Aggregate per-timestep results to per-(Model, Scenario, Region) summary."""
    group_cols = ["Model", "Scenario", "Region", "Scenario_Category"]
    summary = (
        timestep_df.groupby(group_cols)
        .agg(
            n_timesteps=("abs_error", "size"),           # size includes NaN rows, keeping counts consistent
            n_zero_total=("zero_total", "sum"),          # how many timesteps had |total| < 1e-9
            mean_error=("abs_error", "mean"),            # mean skips NaN (zero_total rows excluded)
            max_error=("abs_error", "max"),
            n_failed_timesteps=("passed_timestep", lambda x: (~x).sum()),
        )
        .reset_index()
    )
    summary["passed"] = summary["n_failed_timesteps"] == 0
    summary["mean_error_pct"] = summary["mean_error"] * 100
    summary["max_error_pct"] = summary["max_error"] * 100
    return summary.sort_values("mean_error", ascending=False)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_overview(summary: pd.DataFrame, threshold: float):
    n_total = len(summary)
    n_passed = summary["passed"].sum()
    n_failed = n_total - n_passed
    pass_rate = 100 * n_passed / n_total if n_total > 0 else 0.0

    print(f"\n{'='*60}")
    print("SUM CHECK OVERVIEW")
    print(f"{'='*60}")
    print(f"  Threshold             : {threshold*100:.1f}% error at every timestep")
    print(f"  Total scenarios       : {n_total:,}")
    print(f"  Passed                : {n_passed:,}  ({pass_rate:.1f}%)")
    print(f"  Failed                : {n_failed:,}  ({100-pass_rate:.1f}%)")
    print(f"  Mean error (all)      : {summary['mean_error_pct'].mean():.4f}%")
    print(f"  Median error (all)    : {summary['mean_error_pct'].median():.4f}%")
    print(f"  Max error seen        : {summary['max_error_pct'].max():.4f}%")


def report_by_category(summary: pd.DataFrame):
    print(f"\n{'='*60}")
    print("SUM CHECK BY SCENARIO CATEGORY")
    print(f"{'='*60}")
    cat_summary = (
        summary.groupby("Scenario_Category")
        .agg(
            total=("passed", "count"),
            passed=("passed", "sum"),
            mean_error_pct=("mean_error_pct", "mean"),
            max_error_pct=("max_error_pct", "max"),
        )
        .reset_index()
    )
    cat_summary["pass_rate_%"] = 100 * cat_summary["passed"] / cat_summary["total"]
    cat_summary = cat_summary.sort_values("pass_rate_%")

    for _, row in cat_summary.iterrows():
        print(
            f"  {str(row['Scenario_Category']):<20}  "
            f"passed: {int(row['passed']):>5} / {int(row['total']):>5}  "
            f"({row['pass_rate_%']:5.1f}%)  "
            f"mean error: {row['mean_error_pct']:.4f}%  "
            f"max error: {row['max_error_pct']:.4f}%"
        )


def report_worst(summary: pd.DataFrame, n: int = 15):
    print(f"\n{'='*60}")
    print(f"TOP {n} WORST SCENARIOS (by mean sum-check error)")
    print(f"{'='*60}")
    worst = summary.head(n)
    for _, row in worst.iterrows():
        status = "PASS" if row["passed"] else "FAIL"
        print(
            f"  [{status}]  {row['Model']:<25} | {row['Scenario']:<30} | {row['Region']:<20}  "
            f"mean: {row['mean_error_pct']:.4f}%  max: {row['max_error_pct']:.4f}%  "
            f"failed timesteps: {int(row['n_failed_timesteps'])} / {int(row['n_timesteps'])}"
        )


def report_error_distribution(summary: pd.DataFrame):
    print(f"\n{'='*60}")
    print("MEAN ERROR DISTRIBUTION (across scenarios)")
    print(f"{'='*60}")
    pcts = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    quantiles = np.percentile(summary["mean_error_pct"].dropna(), pcts)
    for p, q in zip(pcts, quantiles):
        bar_len = int(q * 200)  # scale for display
        print(f"  p{p:>3}:  {q:.5f}%")


def report_component_contributions(timestep_df: pd.DataFrame):
    """Show how much each component contributes on average to the total."""
    print(f"\n{'='*60}")
    print("MEAN COMPONENT CONTRIBUTIONS (% of total, across all timesteps)")
    print(f"{'='*60}")
    total_mean = timestep_df["total"].mean()
    for var in COMPONENT_VARS:
        short = var.replace("Secondary Energy|Electricity|", "")
        comp_mean = timestep_df[var].mean()
        share = 100 * comp_mean / total_mean if total_mean != 0 else 0.0
        print(f"  {short:<15}  mean: {comp_mean:>10.3f} EJ/yr  share: {share:>6.2f}%")
    print(f"  {'TOTAL':<15}  mean: {total_mean:>10.3f} EJ/yr")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sum check for Secondary Energy|Electricity")
    parser.add_argument("--run_id", required=True, help="Run ID, e.g. xgb_03")
    parser.add_argument(
        "--threshold", type=float, default=0.012,
        help="Max allowed relative error per timestep (default: 0.012 = 1.2%%)"
    )
    parser.add_argument(
        "--abs_floor", type=float, default=1.0,
        help="Minimum absolute value of total for relative error to be computed. "
             "Timesteps below this are flagged as zero_total and excluded from the metric. "
             "(default: 1.0)"
    )
    parser.add_argument(
        "--use_ground_truth", action="store_true",
        help="Run check against AR6 ground truth (y_test) instead of model predictions. "
             "Useful as a sanity check that the sum constraint holds in the real data."
    )
    args = parser.parse_args()

    # ---- Load ----
    test_data, preds, y_test, targets, required = load_predictions(args.run_id)

    if args.use_ground_truth:
        values = y_test
        label = "AR6 ground truth (y_test)"
        out_suffix = "ground_truth"
    else:
        values = preds
        label = "model predictions"
        out_suffix = ""

    print(f"\n  Data source: {label}")
    long = build_long(test_data, values, targets, required, label)

    # ---- Set up tee ----
    out_subdir = "sum_check_ground_truth" if args.use_ground_truth else "sum_check"
    out_dir = REPO_ROOT / "results" / "xgb" / args.run_id / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.txt"
    sys.stdout = _Tee(report_path)

    # ---- Run check ----
    print(f"\n  Running sum check (threshold = {args.threshold*100:.1f}%, abs_floor = {args.abs_floor})...")
    timestep_df = run_sum_check(long, args.threshold, abs_floor=args.abs_floor)
    summary = scenario_summary(timestep_df, args.threshold)

    print(f"  Timestep rows evaluated: {len(timestep_df):,}")
    print(f"  Scenarios evaluated    : {len(summary):,}")

    # ---- Reports ----
    report_overview(summary, args.threshold)
    report_by_category(summary)
    report_error_distribution(summary)
    report_component_contributions(timestep_df)
    report_worst(summary)

    # ---- Save ----
    summary_path = out_dir / "scenario_summary.csv"
    timestep_path = out_dir / "timestep_errors.csv"
    summary.to_csv(summary_path, index=False)
    timestep_df[
        ["Model", "Scenario", "Region", "Scenario_Category", "Year",
         "total", "sum_components", "zero_total", "abs_error", "passed_timestep"]
    ].to_csv(timestep_path, index=False)

    print(f"\n{'='*60}")
    print(f"Results saved to:")
    print(f"  {report_path}")
    print(f"  {summary_path}")
    print(f"  {timestep_path}")
    print(f"{'='*60}\n")

    if isinstance(sys.stdout, _Tee):
        sys.stdout.close()


if __name__ == "__main__":
    main()