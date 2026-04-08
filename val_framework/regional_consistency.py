"""
Regional consistency check for ML-IAM predictions.

Verifies that predicted values for the World region equal the sum of predicted
values across all subregions in a recognised regional grouping (R5, R6, R10).

For each variable and each (Model, Scenario) that has World + a complete regional
grouping, the check computes:

    error = |World - sum_of_subregions| / |World|

A scenario passes if error < threshold at ALL (variable, year) timesteps where
a complete grouping exists.

Usage:
    python val_framework/regional_consistency.py --run_id xgb_04
    python val_framework/regional_consistency.py --run_id xgb_04 --grouping R10
    python val_framework/regional_consistency.py --run_id xgb_04 --use_ground_truth
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

WORLD_REGION = "World"

# Regions that together sum to World for each aggregation scheme.
# These are the standard AR6 regional groupings.
REGIONAL_GROUPINGS = {
    "R5":  ["R5ASIA", "R5LAM", "R5MAF", "R5OECD90+EU", "R5REF"],
    "R6":  ["R6AFRICA", "R6ASIA", "R6LAM", "R6MIDDLE_EAST", "R6OECD90+EU", "R6REF"],
    "R10": ["R10AFRICA", "R10CHINA+", "R10EUROPE", "R10INDIA+", "R10LATIN_AM",
            "R10MIDDLE_EAST", "R10NORTH_AM", "R10PAC_OECD", "R10REF_ECON",
            "R10REST_ASIA", "R10ROWO"],
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
    """Load and inverse-transform predictions and ground truth."""
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
    preds = y_scaler.inverse_transform(pred_bundle["preds"])
    y_test = y_scaler.inverse_transform(y_test_scaled)

    print(f"  Targets: {len(targets)} variables")
    print(f"  Test set rows: {len(test_data)}")
    return test_data, preds, y_test, targets


def build_long(test_data, values, targets, label):
    """Build long-format DataFrame for all target variables."""
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
# Grouping detection
# ---------------------------------------------------------------------------

def detect_complete_groupings(
    long: pd.DataFrame,
    groupings: dict,
    target_grouping: Optional[str],
) -> dict[tuple, list[str]]:
    """
    For each (Model, Scenario), return which groupings have full region coverage
    (i.e. World + ALL subregions in the grouping are present for that scenario).

    Returns a dict mapping (Model, Scenario) → list of applicable grouping names.
    Only groupings in `target_grouping` are considered, or all if None.
    """
    check_groupings = (
        {target_grouping: groupings[target_grouping]}
        if target_grouping and target_grouping in groupings
        else groupings
    )

    # Get all regions available per (Model, Scenario)
    scenario_regions = (
        long.groupby(["Model", "Scenario"])["Region"]
        .apply(set)
        .to_dict()
    )

    result = {}
    for (model, scenario), regions in scenario_regions.items():
        applicable = []
        for name, subregions in check_groupings.items():
            if WORLD_REGION in regions and all(r in regions for r in subregions):
                applicable.append(name)
        if applicable:
            result[(model, scenario)] = applicable

    return result


# ---------------------------------------------------------------------------
# Consistency check
# ---------------------------------------------------------------------------

def run_regional_check(
    long: pd.DataFrame,
    applicable: dict,
    groupings: dict,
    threshold: float,
    abs_floor: float = 1.0,
) -> pd.DataFrame:
    """
    For each (Model, Scenario, Variable, Year) where a complete grouping exists,
    compare the World value to the sum of subregion values.

    Returns a timestep-level DataFrame with columns:
        Model, Scenario, Variable, Scenario_Category, Year, grouping,
        world_value, sum_subregions, zero_world, abs_error, passed_timestep
    """
    if not applicable:
        print("  No (Model, Scenario) pairs have a complete regional grouping. Skipping.")
        return pd.DataFrame()

    # Collect all subregion lists needed
    all_subregions = set()
    for subregions_list in groupings.values():
        all_subregions.update(subregions_list)
    all_subregions.add(WORLD_REGION)

    # Restrict to relevant scenarios and regions
    scen_filter = pd.Series(
        [f"{r[0]}||{r[1]}" for r in applicable.keys()]
    )
    long["_key"] = long["Model"] + "||" + long["Scenario"]
    subset = long[
        long["_key"].isin(scen_filter.values) &
        long["Region"].isin(all_subregions)
    ].drop(columns="_key").copy()

    rows_list = []

    for (model, scenario), grouping_names in applicable.items():
        scen_data = subset[
            (subset["Model"] == model) & (subset["Scenario"] == scenario)
        ]
        # Get Scenario_Category (same for all rows of this scenario)
        cat = scen_data["Scenario_Category"].iloc[0]

        for grouping_name in grouping_names:
            subregions = groupings[grouping_name]

            # Pivot so we have one column per region
            try:
                pivot = scen_data.pivot_table(
                    index=["Variable", "Year"],
                    columns="Region",
                    values="Value",
                    aggfunc="first",
                ).reset_index()
            except Exception:
                continue

            if WORLD_REGION not in pivot.columns:
                continue
            missing_subs = [r for r in subregions if r not in pivot.columns]
            if missing_subs:
                continue

            pivot["sum_subregions"] = pivot[subregions].sum(axis=1)
            pivot["world_value"]    = pivot[WORLD_REGION]
            pivot["grouping"]       = grouping_name
            pivot["Model"]          = model
            pivot["Scenario"]       = scenario
            pivot["Scenario_Category"] = cat

            pivot["zero_world"] = pivot["world_value"].abs() < abs_floor
            pivot["abs_error"]  = np.where(
                pivot["zero_world"],
                np.nan,
                (pivot["world_value"] - pivot["sum_subregions"]).abs() / pivot["world_value"].abs(),
            )
            pivot["passed_timestep"] = pivot["abs_error"] < threshold

            keep = ["Model", "Scenario", "Scenario_Category", "Variable", "Year",
                    "grouping", "world_value", "sum_subregions",
                    "zero_world", "abs_error", "passed_timestep"]
            rows_list.append(pivot[[c for c in keep if c in pivot.columns]])

    if not rows_list:
        return pd.DataFrame()

    long.drop(columns=["_key"], errors="ignore", inplace=True)
    return pd.concat(rows_list, ignore_index=True)


def scenario_summary(timestep_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-timestep results to per-(Model, Scenario, Variable, grouping)."""
    group_cols = ["Model", "Scenario", "Scenario_Category", "Variable", "grouping"]
    summary = (
        timestep_df.groupby(group_cols)
        .agg(
            n_timesteps        =("abs_error", "size"),
            n_zero_world       =("zero_world", "sum"),
            mean_error         =("abs_error", "mean"),
            max_error          =("abs_error", "max"),
            n_failed_timesteps =("passed_timestep", lambda x: (~x).sum()),
        )
        .reset_index()
    )
    summary["passed"]         = summary["n_failed_timesteps"] == 0
    summary["mean_error_pct"] = summary["mean_error"] * 100
    summary["max_error_pct"]  = summary["max_error"] * 100
    return summary.sort_values(["grouping", "Variable", "mean_error"],
                               ascending=[True, True, False])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_overview(summary: pd.DataFrame, threshold: float, applicable: dict):
    n_total  = len(summary)
    n_passed = summary["passed"].sum()
    pass_rate = 100 * n_passed / n_total if n_total > 0 else 0.0
    n_scenarios = len(applicable)

    print(f"\n{'='*60}")
    print("REGIONAL CONSISTENCY CHECK OVERVIEW")
    print(f"{'='*60}")
    print(f"  Threshold                    : {threshold*100:.1f}% error at every timestep")
    print(f"  Scenarios with full grouping : {n_scenarios:,}")
    print(f"  Total (scenario × variable)  : {n_total:,}")
    print(f"  Passed                       : {n_passed:,}  ({pass_rate:.1f}%)")
    print(f"  Failed                       : {n_total - n_passed:,}  ({100-pass_rate:.1f}%)")
    if n_total > 0:
        print(f"  Mean error (all)             : {summary['mean_error_pct'].mean():.4f}%")
        print(f"  Max error seen               : {summary['max_error_pct'].max():.4f}%")


def report_by_grouping(summary: pd.DataFrame):
    print(f"\n{'='*60}")
    print("RESULTS BY REGIONAL GROUPING")
    print(f"{'='*60}")
    for grouping, grp in summary.groupby("grouping"):
        n_total  = len(grp)
        n_passed = grp["passed"].sum()
        pass_rate = 100 * n_passed / n_total if n_total > 0 else 0.0
        print(
            f"\n  {grouping}  —  "
            f"passed: {n_passed:,} / {n_total:,}  ({pass_rate:.1f}%)  "
            f"mean error: {grp['mean_error_pct'].mean():.4f}%  "
            f"max error: {grp['max_error_pct'].max():.4f}%"
        )


def report_by_variable(summary: pd.DataFrame):
    print(f"\n{'='*60}")
    print("RESULTS BY VARIABLE")
    print(f"{'='*60}")
    var_summary = (
        summary.groupby(["grouping", "Variable"])
        .agg(
            total          =("passed", "count"),
            passed         =("passed", "sum"),
            mean_error_pct =("mean_error_pct", "mean"),
            max_error_pct  =("max_error_pct", "max"),
        )
        .reset_index()
    )
    var_summary["pass_rate_%"] = 100 * var_summary["passed"] / var_summary["total"]
    for grouping, grp in var_summary.groupby("grouping"):
        print(f"\n  {grouping}")
        for _, row in grp.sort_values("pass_rate_%").iterrows():
            print(
                f"    {row['Variable']:<35}  "
                f"passed: {int(row['passed']):>5} / {int(row['total']):>5}  "
                f"({row['pass_rate_%']:5.1f}%)  "
                f"mean error: {row['mean_error_pct']:.4f}%  "
                f"max error: {row['max_error_pct']:.4f}%"
            )


def report_worst(summary: pd.DataFrame, n: int = 15):
    print(f"\n{'='*60}")
    print(f"TOP {n} WORST (scenario × variable) BY MEAN ERROR")
    print(f"{'='*60}")
    worst = summary.sort_values("mean_error", ascending=False).head(n)
    for _, row in worst.iterrows():
        status = "PASS" if row["passed"] else "FAIL"
        print(
            f"  [{status}]  [{row['grouping']}]  {row['Variable']:<35}  "
            f"{row['Model']:<20} | {row['Scenario']:<25}  "
            f"mean: {row['mean_error_pct']:.4f}%  max: {row['max_error_pct']:.4f}%"
        )


def report_coverage(applicable: dict, groupings: dict):
    print(f"\n{'='*60}")
    print("COVERAGE: SCENARIOS WITH COMPLETE REGIONAL GROUPINGS")
    print(f"{'='*60}")
    counts = {}
    for grouping_names in applicable.values():
        for g in grouping_names:
            counts[g] = counts.get(g, 0) + 1
    for g in sorted(groupings.keys()):
        n = counts.get(g, 0)
        print(f"  {g}:  {n} scenario(s) with all {len(groupings[g])} subregions present")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Regional consistency check for ML-IAM predictions"
    )
    parser.add_argument("--run_id", required=True, help="Run ID, e.g. xgb_04")
    parser.add_argument(
        "--threshold", type=float, default=0.012,
        help="Max allowed relative error per timestep (default: 0.012 = 1.2%%)"
    )
    parser.add_argument(
        "--abs_floor", type=float, default=1.0,
        help="Minimum |World value| for relative error to be computed. (default: 1.0)"
    )
    parser.add_argument(
        "--grouping", type=str, default=None, choices=list(REGIONAL_GROUPINGS.keys()),
        help="Restrict check to a single grouping (default: check all)"
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
        out_subdir = "regional_consistency_ground_truth"
    else:
        values     = preds
        label      = "model predictions"
        out_subdir = "regional_consistency"

    print(f"\n  Data source: {label}")
    long = build_long(test_data, values, targets, label)

    # ---- Detect applicable groupings ----
    print(f"\n  Detecting complete regional groupings...")
    applicable = detect_complete_groupings(long, REGIONAL_GROUPINGS, args.grouping)

    groupings_to_check = (
        {args.grouping: REGIONAL_GROUPINGS[args.grouping]}
        if args.grouping
        else REGIONAL_GROUPINGS
    )

    print(f"  Scenarios with at least one complete grouping: {len(applicable)}")
    for g, subregions in groupings_to_check.items():
        n = sum(1 for v in applicable.values() if g in v)
        print(f"    {g}: {n} scenario(s)  (requires {len(subregions)} subregions)")

    # ---- Set up output dir and tee ----
    out_dir = REPO_ROOT / "results" / "xgb" / args.run_id / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(out_dir / "report.txt")

    # ---- Run check ----
    print(f"\n  Running regional consistency check "
          f"(threshold={args.threshold*100:.1f}%, abs_floor={args.abs_floor})...")
    timestep_df = run_regional_check(
        long, applicable, groupings_to_check, args.threshold, args.abs_floor
    )

    if timestep_df.empty:
        print("\n  No regional consistency checks could be run.")
        print("  This is expected if the model was not trained on multi-region scenarios.")
        if isinstance(sys.stdout, _Tee):
            sys.stdout.close()
        return

    summary = scenario_summary(timestep_df)
    print(f"  Timestep rows evaluated    : {len(timestep_df):,}")
    print(f"  Scenario × variable checks : {len(summary):,}")

    # ---- Reports ----
    report_coverage(applicable, groupings_to_check)
    report_overview(summary, args.threshold, applicable)
    report_by_grouping(summary)
    report_by_variable(summary)
    report_worst(summary)

    # ---- Save ----
    summary_path  = out_dir / "scenario_summary.csv"
    timestep_path = out_dir / "timestep_errors.csv"
    summary.to_csv(summary_path, index=False)
    timestep_df.to_csv(timestep_path, index=False)

    print(f"\n{'='*60}")
    print(f"Results saved to:")
    print(f"  {out_dir / 'report.txt'}")
    print(f"  {summary_path}")
    print(f"  {timestep_path}")
    print(f"{'='*60}\n")

    if isinstance(sys.stdout, _Tee):
        sys.stdout.close()


if __name__ == "__main__":
    main()
