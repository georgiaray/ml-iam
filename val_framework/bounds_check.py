"""
Physical bounds check for ML-IAM predictions.

Checks predicted values against two types of bounds:

  1. Hard physical bounds — absolute constraints that no physically plausible
     scenario can violate (e.g. energy generation cannot be negative). These
     are defined in PHYSICAL_BOUNDS below.

  2. Empirical bounds (on by default, disable with --no_empirical) — per-variable
     (p_lo, p_hi) percentile range derived from the AR6 test-set ground truth.
     These represent the envelope of plausible behaviour seen in real IAM data.
     The --percentile flag controls the tail size (default 1%).

A timestep is flagged if it violates either type of bound.

Usage:
    python val_framework/bounds_check.py --run_id xgb_04
    python val_framework/bounds_check.py --run_id xgb_04 --no_empirical
    python val_framework/bounds_check.py --run_id xgb_04 --percentile 5
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Hard physical bounds
# Each entry is (lower_bound, upper_bound); None means no constraint that side.
# Energy and capacity variables must be non-negative (physical law).
# Emissions can go slightly negative in CDR scenarios, so no hard lower bound.
# ---------------------------------------------------------------------------

PHYSICAL_BOUNDS: dict = {
    "Primary Energy|Coal":                    (0.0, None),
    "Primary Energy|Gas":                     (0.0, None),
    "Primary Energy|Nuclear":                 (0.0, None),
    "Primary Energy|Oil":                     (0.0, None),
    "Primary Energy|Solar":                   (0.0, None),
    "Primary Energy|Wind":                    (0.0, None),
    "Secondary Energy|Electricity":           (0.0, None),
    "Secondary Energy|Electricity|Biomass":   (0.0, None),
    "Secondary Energy|Electricity|Coal":      (0.0, None),
    "Secondary Energy|Electricity|Gas":       (0.0, None),
    "Secondary Energy|Electricity|Geothermal":(0.0, None),
    "Secondary Energy|Electricity|Hydro":     (0.0, None),
    "Secondary Energy|Electricity|Nuclear":   (0.0, None),
    "Secondary Energy|Electricity|Oil":       (0.0, None),
    "Secondary Energy|Electricity|Solar":     (0.0, None),
    "Secondary Energy|Electricity|Wind":      (0.0, None),
}


# ---------------------------------------------------------------------------
# Tee
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
    from src.utils.run_store import RunStore
    from scripts.train_xgb import derive_splits

    print(f"\n{'='*60}")
    print(f"Loading artifacts for run: {run_id}")
    print(f"{'='*60}")

    store  = RunStore(run_id)
    data   = store.load_processed_data()
    splits = derive_splits(data)

    test_data     = splits["test_data"]
    y_test_scaled = splits["y_test"]
    targets       = splits["targets"]
    y_scaler      = splits["y_scaler"]

    pred_bundle = store.load_predictions()
    preds  = y_scaler.inverse_transform(pred_bundle["preds"])
    y_test = y_scaler.inverse_transform(y_test_scaled)

    print(f"  Targets: {len(targets)} variables")
    return test_data, preds, y_test, targets


def build_long(test_data, values, targets, label):
    index_cols = ["Model", "Scenario", "Region", "Scenario_Category", "Year"]
    idx  = test_data[index_cols].reset_index(drop=True)
    wide = pd.DataFrame(values, columns=targets)
    combined = pd.concat([idx, wide], axis=1)
    long = combined.melt(
        id_vars=index_cols,
        value_vars=targets,
        var_name="Variable",
        value_name="Value",
    )
    return long


# ---------------------------------------------------------------------------
# Bounds derivation
# ---------------------------------------------------------------------------

def derive_empirical_bounds(
    gt_long: pd.DataFrame,
    targets: list,
    lower_pct: float,
    upper_pct: float,
) -> pd.DataFrame:
    """
    Compute per-variable (lower_pct, upper_pct) percentile bounds from the
    AR6 test-set ground truth. Returns a DataFrame with columns:
        Variable, empirical_lower, empirical_upper
    """
    stats = (
        gt_long.groupby("Variable")["Value"]
        .describe(percentiles=[lower_pct / 100, upper_pct / 100])
        .reset_index()
    )
    lo_col = next(c for c in stats.columns if c.endswith("%") and float(c[:-1]) == lower_pct)
    hi_col = next(c for c in stats.columns if c.endswith("%") and float(c[:-1]) == upper_pct)
    return stats[["Variable", lo_col, hi_col]].rename(
        columns={lo_col: "empirical_lower", hi_col: "empirical_upper"}
    )


def build_bounds_table(
    targets: list,
    use_empirical: bool,
    empirical_bounds: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine hard physical bounds and empirical bounds (if requested) into a
    single DataFrame with columns:
        Variable, lower_bound, upper_bound
    where lower_bound = max(physical_lower, empirical_lower) if both exist,
    and upper_bound = min(physical_upper, empirical_upper) if both exist.
    """
    rows = []
    for var in targets:
        phys_lo, phys_hi = PHYSICAL_BOUNDS.get(var, (None, None))

        emp_lo, emp_hi = None, None
        if use_empirical and empirical_bounds is not None:
            row = empirical_bounds[empirical_bounds["Variable"] == var]
            if not row.empty:
                emp_lo = row["empirical_lower"].iloc[0]
                emp_hi = row["empirical_upper"].iloc[0]

        # Take the more restrictive of the two lower bounds
        if phys_lo is not None and emp_lo is not None:
            lower = max(phys_lo, emp_lo)
        else:
            lower = phys_lo if phys_lo is not None else emp_lo

        # Take the more restrictive of the two upper bounds
        if phys_hi is not None and emp_hi is not None:
            upper = min(phys_hi, emp_hi)
        else:
            upper = phys_hi if phys_hi is not None else emp_hi

        if lower is not None or upper is not None:
            rows.append({
                "Variable":    var,
                "lower_bound": lower,
                "upper_bound": upper,
                "has_physical_lower": phys_lo is not None,
                "has_physical_upper": phys_hi is not None,
                "has_empirical_lower": emp_lo is not None,
                "has_empirical_upper": emp_hi is not None,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bounds check
# ---------------------------------------------------------------------------

def run_bounds_check(
    pred_long: pd.DataFrame,
    bounds_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flag every timestep in pred_long where the predicted value falls outside
    the bounds defined in bounds_table.

    Returns a DataFrame of all checked rows with columns:
        ...(original columns)..., lower_bound, upper_bound,
        below_lower, above_upper, violation
    """
    checked = pred_long.merge(
        bounds_table[["Variable", "lower_bound", "upper_bound"]],
        on="Variable",
        how="inner",  # only variables that have at least one bound
    )

    checked["below_lower"] = (
        checked["lower_bound"].notna() &
        (checked["Value"] < checked["lower_bound"])
    )
    checked["above_upper"] = (
        checked["upper_bound"].notna() &
        (checked["Value"] > checked["upper_bound"])
    )
    checked["violation"] = checked["below_lower"] | checked["above_upper"]

    return checked


def scenario_summary(checked: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to per-(Model, Scenario, Region, Variable) summaries."""
    group_cols = ["Model", "Scenario", "Region", "Scenario_Category", "Variable"]
    summary = (
        checked.groupby(group_cols)
        .agg(
            n_timesteps   =("violation", "count"),
            n_violations  =("violation", "sum"),
            n_below_lower =("below_lower", "sum"),
            n_above_upper =("above_upper", "sum"),
            min_value     =("Value", "min"),
            max_value     =("Value", "max"),
            lower_bound   =("lower_bound", "first"),
            upper_bound   =("upper_bound", "first"),
        )
        .reset_index()
    )
    summary["passed"]         = summary["n_violations"] == 0
    summary["violation_rate"] = summary["n_violations"] / summary["n_timesteps"]
    return summary.sort_values("n_violations", ascending=False)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_bounds_table(bounds_table: pd.DataFrame):
    print(f"\n{'='*60}")
    print("BOUNDS IN USE")
    print(f"{'='*60}")
    for _, row in bounds_table.iterrows():
        lo = f"{row['lower_bound']:.4g}" if pd.notna(row["lower_bound"]) else "—"
        hi = f"{row['upper_bound']:.4g}" if pd.notna(row["upper_bound"]) else "—"
        sources = []
        if row.get("has_physical_lower") or row.get("has_physical_upper"):
            sources.append("physical")
        if row.get("has_empirical_lower") or row.get("has_empirical_upper"):
            sources.append("empirical")
        print(f"  {row['Variable']:<40}  [{lo:>10}, {hi:>10}]  ({', '.join(sources)})")


def report_overview(checked: pd.DataFrame, summary: pd.DataFrame):
    n_rows    = len(checked)
    n_viol    = checked["violation"].sum()
    viol_rate = 100 * n_viol / n_rows if n_rows > 0 else 0.0
    n_scen    = len(summary)
    n_clean   = summary["passed"].sum()

    print(f"\n{'='*60}")
    print("BOUNDS CHECK OVERVIEW")
    print(f"{'='*60}")
    print(f"  Timesteps checked        : {n_rows:,}")
    print(f"  Violations               : {n_viol:,}  ({viol_rate:.3f}%)")
    print(f"  Below lower bound        : {checked['below_lower'].sum():,}")
    print(f"  Above upper bound        : {checked['above_upper'].sum():,}")
    print(f"  Scenario-regions checked : {n_scen:,}")
    print(f"  Fully clean              : {n_clean:,}  ({100*n_clean/n_scen:.1f}%)")


def report_by_variable(summary: pd.DataFrame):
    print(f"\n{'='*60}")
    print("VIOLATIONS BY VARIABLE")
    print(f"{'='*60}")
    var_summary = (
        summary.groupby("Variable")
        .agg(
            total         =("n_violations", "count"),
            n_violations  =("n_violations", "sum"),
            n_below_lower =("n_below_lower", "sum"),
            n_above_upper =("n_above_upper", "sum"),
            min_value     =("min_value", "min"),
            lower_bound   =("lower_bound", "first"),
        )
        .reset_index()
    )
    var_summary = var_summary.sort_values("n_violations", ascending=False)
    for _, row in var_summary.iterrows():
        if row["n_violations"] == 0:
            continue
        lo = f"{row['lower_bound']:.4g}" if pd.notna(row["lower_bound"]) else "—"
        print(
            f"  {row['Variable']:<40}  "
            f"violations: {int(row['n_violations']):>6}  "
            f"(below: {int(row['n_below_lower'])}, above: {int(row['n_above_upper'])})  "
            f"min value seen: {row['min_value']:.4g}  lower bound: {lo}"
        )
    if var_summary["n_violations"].sum() == 0:
        print("  No violations detected.")


def report_by_category(summary: pd.DataFrame):
    print(f"\n{'='*60}")
    print("VIOLATIONS BY SCENARIO CATEGORY")
    print(f"{'='*60}")
    cat_summary = (
        summary.groupby("Scenario_Category")
        .agg(total=("n_violations", "count"), violations=("n_violations", "sum"))
        .reset_index()
    )
    cat_summary["rate_%"] = 100 * cat_summary["violations"] / cat_summary["total"]
    cat_summary = cat_summary.sort_values("rate_%", ascending=False)
    for _, row in cat_summary.iterrows():
        print(
            f"  {str(row['Scenario_Category']):<20}  "
            f"violations: {int(row['violations']):>6}  ({row['rate_%']:.2f}%)"
        )


def report_worst(summary: pd.DataFrame, n: int = 15):
    print(f"\n{'='*60}")
    print(f"TOP {n} WORST SCENARIO-REGIONS BY VIOLATION COUNT")
    print(f"{'='*60}")
    worst = summary[summary["n_violations"] > 0].head(n)
    if worst.empty:
        print("  No violations.")
        return
    for _, row in worst.iterrows():
        lo = f"{row['lower_bound']:.4g}" if pd.notna(row["lower_bound"]) else "—"
        hi = f"{row['upper_bound']:.4g}" if pd.notna(row["upper_bound"]) else "—"
        print(
            f"  {row['Variable']:<35}  "
            f"{row['Model']:<20} | {row['Scenario']:<25} | {row['Region']:<15}  "
            f"violations: {int(row['n_violations'])} / {int(row['n_timesteps'])}  "
            f"min: {row['min_value']:.4g}  max: {row['max_value']:.4g}  "
            f"bounds: [{lo}, {hi}]"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Physical bounds check for ML-IAM predictions"
    )
    parser.add_argument("--run_id", required=True, help="Run ID, e.g. xgb_04")
    parser.add_argument(
        "--no_empirical", action="store_true",
        help="Disable empirical per-variable bounds derived from the AR6 "
             "test-set ground truth. By default, empirical bounds are applied."
    )
    parser.add_argument(
        "--percentile", type=float, default=1.0,
        help="Tail percentile for empirical bounds (default: 1.0). "
             "Has no effect if --no_empirical is set."
    )
    parser.add_argument(
        "--use_ground_truth", action="store_true",
        help="Check AR6 ground truth (y_test) instead of model predictions."
    )
    args = parser.parse_args()

    # ---- Load ----
    test_data, preds, y_test, targets = load_predictions(args.run_id)

    if args.use_ground_truth:
        values     = y_test
        label      = "AR6 ground truth (y_test)"
        out_subdir = "bounds_check_ground_truth"
    else:
        values     = preds
        label      = "model predictions"
        out_subdir = "bounds_check"

    print(f"\n  Data source: {label}")

    gt_long   = build_long(test_data, y_test,  targets, "ground_truth")
    pred_long = build_long(test_data, values,   targets, label)

    # ---- Build bounds table ----
    use_empirical = not args.no_empirical
    empirical_bounds = None
    if use_empirical:
        lower_pct = args.percentile
        upper_pct = 100.0 - args.percentile
        print(f"\n  Deriving empirical bounds from AR6 ground truth "
              f"({lower_pct}th / {upper_pct}th percentiles)...")
        empirical_bounds = derive_empirical_bounds(gt_long, targets, lower_pct, upper_pct)
    else:
        print("\n  Empirical bounds disabled (--no_empirical). Using physical bounds only.")

    bounds_table = build_bounds_table(targets, use_empirical, empirical_bounds)

    vars_with_bounds = set(bounds_table["Variable"])
    vars_without = [v for v in targets if v not in vars_with_bounds]
    print(f"\n  Variables with at least one bound : {len(bounds_table)}")
    if vars_without:
        print(f"  Variables with no bounds (skipped): {len(vars_without)}")
        for v in vars_without:
            print(f"    - {v}")

    # ---- Set up output dir and tee ----
    out_dir = REPO_ROOT / "results" / "xgb" / args.run_id / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(out_dir / "report.txt")

    # ---- Run check ----
    print(f"\n  Running bounds check...")
    checked = run_bounds_check(pred_long, bounds_table)
    summary = scenario_summary(checked)

    # ---- Reports ----
    report_bounds_table(bounds_table)
    report_overview(checked, summary)
    report_by_variable(summary)
    report_by_category(summary)
    report_worst(summary)

    # ---- Save ----
    summary_path   = out_dir / "scenario_summary.csv"
    violations_path = out_dir / "violations.csv"
    bounds_path    = out_dir / "bounds_used.csv"

    summary.to_csv(summary_path, index=False)
    checked[checked["violation"]].to_csv(violations_path, index=False)
    bounds_table.to_csv(bounds_path, index=False)

    print(f"\n{'='*60}")
    print(f"Results saved to:")
    print(f"  {out_dir / 'report.txt'}")
    print(f"  {summary_path}")
    print(f"  {violations_path}  (violations only)")
    print(f"  {bounds_path}")
    print(f"{'='*60}\n")

    if isinstance(sys.stdout, _Tee):
        sys.stdout.close()


if __name__ == "__main__":
    main()
