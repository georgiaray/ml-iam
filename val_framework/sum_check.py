"""
Hierarchy sum check for ML-IAM predictions.

Automatically discovers all parent-child variable relationships in the model's
target set (using the `|` separator convention) and verifies that each parent
variable equals the sum of its direct children at every timestep.

Error metric (per timestep):
    e = |parent - sum_of_children| / |parent|

A scenario passes if e < threshold at ALL timesteps for ALL parent variables.

Usage:
    python val_framework/sum_check.py --run_id xgb_04
    python val_framework/sum_check.py --run_id xgb_04 --threshold 0.05
    python val_framework/sum_check.py --run_id xgb_04 --use_ground_truth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Hierarchy discovery
# ---------------------------------------------------------------------------

def discover_hierarchy(variables: list[str]) -> dict[str, list[str]]:
    """
    Find all parent-child relationships in a list of variables using the
    `|` separator convention. A variable is a parent if at least one other
    variable is exactly one level below it (i.e. `parent + "|" + leaf`).

    Returns a dict mapping each parent to its list of direct children.
    Only parent variables that are themselves present in `variables` are
    included (so we can compare the predicted parent against the sum).

    Example
    -------
    Given ["A", "A|B", "A|C", "A|B|D"], returns:
        {"A": ["A|B", "A|C"], "A|B": ["A|B|D"]}
    """
    var_set = set(variables)
    hierarchy = {}
    for var in variables:
        prefix = var + "|"
        children = [
            v for v in variables
            if v.startswith(prefix) and "|" not in v[len(prefix):]
        ]
        if children:
            hierarchy[var] = children
    return hierarchy


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

def load_predictions(run_id: str, hierarchy: dict[str, list[str]]):
    """
    Load predictions and ground truth for all variables in the hierarchy.
    Returns (test_data, preds, y_test, targets).
    """
    from src.utils.run_store import RunStore
    from scripts.train_xgb import derive_splits

    print(f"\n{'='*60}")
    print(f"Loading artifacts for run: {run_id}")
    print(f"{'='*60}")

    store = RunStore(run_id)
    data = store.load_processed_data()
    splits = derive_splits(data)

    test_data      = splits["test_data"]
    y_test_scaled  = splits["y_test"]
    targets        = splits["targets"]
    y_scaler       = splits["y_scaler"]

    # Check that every variable in the hierarchy is in targets
    required = set(hierarchy.keys()) | {c for children in hierarchy.values() for c in children}
    missing = required - set(targets)
    if missing:
        print("\nERROR: The following hierarchy variables are missing from this run's targets:")
        for v in sorted(missing):
            print(f"  - {v}")
        sys.exit(1)

    print("  Inverse-transforming predictions and ground truth...")
    pred_bundle  = store.load_predictions()
    preds        = y_scaler.inverse_transform(pred_bundle["preds"])
    y_test       = y_scaler.inverse_transform(y_test_scaled)

    print(f"  Targets: {len(targets)} variables")
    print(f"  Hierarchy parents found: {list(hierarchy.keys())}")
    return test_data, preds, y_test, targets


def build_long(test_data, values, targets, label):
    """Build long-format DataFrame from a values array (all target variables)."""
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
# Sum check (per parent variable)
# ---------------------------------------------------------------------------

def run_sum_check(
    long: pd.DataFrame,
    parent: str,
    children: list[str],
    threshold: float,
    abs_floor: float = 1.0,
) -> pd.DataFrame:
    """
    For each (Model, Scenario, Region, Year), compute:
        sum_children  = sum of `children` variables
        parent_value  = value of `parent` variable
        error         = |parent_value - sum_children| / |parent_value|

    Returns a timestep-level DataFrame with columns:
        parent_variable, total, sum_components, zero_total, abs_error, passed_timestep

    Timesteps where |parent_value| < abs_floor are flagged as zero_total=True and
    abs_error is set to NaN. These count as failed timesteps (passed_timestep=False)
    and are visible via n_zero_total in scenario_summary.
    """
    all_vars = [parent] + children
    subset = long[long["Variable"].isin(all_vars)].copy()

    index_cols = ["Model", "Scenario", "Region", "Scenario_Category", "Year", "Variable"]
    dupes = subset[subset.duplicated(subset=index_cols)]
    if not dupes.empty:
        raise ValueError(
            f"{len(dupes)} duplicate rows found for parent '{parent}' "
            f"(e.g. {dupes[index_cols].iloc[0].to_dict()}). "
            "Check your splits/data loading pipeline."
        )

    pivot = subset.pivot_table(
        index=["Model", "Scenario", "Region", "Scenario_Category", "Year"],
        columns="Variable",
        values="Value",
        aggfunc="first",
    ).reset_index()

    pivot["sum_components"] = pivot[children].sum(axis=1)
    pivot["total"]          = pivot[parent]
    pivot["parent_variable"] = parent

    pivot["zero_total"] = pivot["total"].abs() < abs_floor
    pivot["abs_error"]  = np.where(
        pivot["zero_total"],
        np.nan,
        (pivot["total"] - pivot["sum_components"]).abs() / pivot["total"].abs(),
    )
    pivot["passed_timestep"] = pivot["abs_error"] < threshold

    keep = ["Model", "Scenario", "Region", "Scenario_Category", "Year",
            "parent_variable", "total", "sum_components", "zero_total",
            "abs_error", "passed_timestep"]
    return pivot[keep]


def scenario_summary(timestep_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Aggregate per-timestep results to per-(Model, Scenario, Region, parent_variable)."""
    group_cols = ["Model", "Scenario", "Region", "Scenario_Category", "parent_variable"]
    summary = (
        timestep_df.groupby(group_cols)
        .agg(
            n_timesteps         =("abs_error", "size"),
            n_zero_total        =("zero_total", "sum"),
            mean_error          =("abs_error", "mean"),
            max_error           =("abs_error", "max"),
            n_failed_timesteps  =("passed_timestep", lambda x: (~x).sum()),
        )
        .reset_index()
    )
    summary["passed"]         = summary["n_failed_timesteps"] == 0
    summary["mean_error_pct"] = summary["mean_error"] * 100
    summary["max_error_pct"]  = summary["max_error"] * 100
    return summary.sort_values(["parent_variable", "mean_error"], ascending=[True, False])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_overview(summary: pd.DataFrame, threshold: float):
    n_total  = len(summary)
    n_passed = summary["passed"].sum()
    pass_rate = 100 * n_passed / n_total if n_total > 0 else 0.0

    print(f"\n{'='*60}")
    print("SUM CHECK OVERVIEW (all parent variables combined)")
    print(f"{'='*60}")
    print(f"  Threshold             : {threshold*100:.1f}% error at every timestep")
    print(f"  Total scenario-regions: {n_total:,}")
    print(f"  Passed                : {n_passed:,}  ({pass_rate:.1f}%)")
    print(f"  Failed                : {n_total - n_passed:,}  ({100-pass_rate:.1f}%)")
    print(f"  Mean error (all)      : {summary['mean_error_pct'].mean():.4f}%")
    print(f"  Median error (all)    : {summary['mean_error_pct'].median():.4f}%")
    print(f"  Max error seen        : {summary['max_error_pct'].max():.4f}%")


def report_by_parent(summary: pd.DataFrame, threshold: float):
    print(f"\n{'='*60}")
    print("SUM CHECK BY PARENT VARIABLE")
    print(f"{'='*60}")
    for parent, grp in summary.groupby("parent_variable"):
        n_total  = len(grp)
        n_passed = grp["passed"].sum()
        pass_rate = 100 * n_passed / n_total if n_total > 0 else 0.0
        print(
            f"\n  {parent}")
        print(
            f"    passed: {n_passed:>6,} / {n_total:>6,}  ({pass_rate:.1f}%)  "
            f"mean error: {grp['mean_error_pct'].mean():.4f}%  "
            f"max error: {grp['max_error_pct'].max():.4f}%"
        )


def report_by_category(summary: pd.DataFrame):
    print(f"\n{'='*60}")
    print("SUM CHECK BY SCENARIO CATEGORY")
    print(f"{'='*60}")
    cat_summary = (
        summary.groupby(["parent_variable", "Scenario_Category"])
        .agg(
            total          =("passed", "count"),
            passed         =("passed", "sum"),
            mean_error_pct =("mean_error_pct", "mean"),
            max_error_pct  =("max_error_pct", "max"),
        )
        .reset_index()
    )
    cat_summary["pass_rate_%"] = 100 * cat_summary["passed"] / cat_summary["total"]

    for parent, grp in cat_summary.groupby("parent_variable"):
        print(f"\n  {parent}")
        for _, row in grp.sort_values("pass_rate_%").iterrows():
            print(
                f"    {str(row['Scenario_Category']):<20}  "
                f"passed: {int(row['passed']):>5} / {int(row['total']):>5}  "
                f"({row['pass_rate_%']:5.1f}%)  "
                f"mean error: {row['mean_error_pct']:.4f}%  "
                f"max error: {row['max_error_pct']:.4f}%"
            )


def report_worst(summary: pd.DataFrame, n: int = 15):
    print(f"\n{'='*60}")
    print(f"TOP {n} WORST SCENARIO-REGIONS (by mean error)")
    print(f"{'='*60}")
    worst = summary.sort_values("mean_error", ascending=False).head(n)
    for _, row in worst.iterrows():
        status = "PASS" if row["passed"] else "FAIL"
        print(
            f"  [{status}]  {row['parent_variable']:<35}  "
            f"{row['Model']:<20} | {row['Scenario']:<25} | {row['Region']:<15}  "
            f"mean: {row['mean_error_pct']:.4f}%  max: {row['max_error_pct']:.4f}%  "
            f"failed timesteps: {int(row['n_failed_timesteps'])} / {int(row['n_timesteps'])}"
        )


def report_error_distribution(summary: pd.DataFrame):
    print(f"\n{'='*60}")
    print("MEAN ERROR DISTRIBUTION (across all scenario-regions)")
    print(f"{'='*60}")
    pcts = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    quantiles = np.percentile(summary["mean_error_pct"].dropna(), pcts)
    for p, q in zip(pcts, quantiles):
        print(f"  p{p:>3}:  {q:.5f}%")


def report_component_contributions(timestep_df: pd.DataFrame, hierarchy: dict):
    print(f"\n{'='*60}")
    print("MEAN COMPONENT CONTRIBUTIONS (% of parent, across all timesteps)")
    print(f"{'='*60}")
    for parent, children in hierarchy.items():
        sub = timestep_df[timestep_df["parent_variable"] == parent]
        print(f"\n  {parent}")
        # Rebuild the wide form from the original long data is not available here;
        # we report what we have (total and sum_components)
        total_mean = sub["total"].mean()
        sum_mean   = sub["sum_components"].mean()
        gap_pct    = 100 * (total_mean - sum_mean) / total_mean if total_mean != 0 else 0.0
        print(f"    mean parent value    : {total_mean:>12.3f}")
        print(f"    mean sum of children : {sum_mean:>12.3f}")
        print(f"    mean gap (parent - sum): {gap_pct:>+.4f}%  "
              f"({'children over-predict' if gap_pct < 0 else 'children under-predict'})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hierarchy sum check for ML-IAM predictions")
    parser.add_argument("--run_id", required=True, help="Run ID, e.g. xgb_04")
    parser.add_argument(
        "--threshold", type=float, default=0.012,
        help="Max allowed relative error per timestep (default: 0.012 = 1.2%%)"
    )
    parser.add_argument(
        "--abs_floor", type=float, default=1.0,
        help="Minimum |parent value| for relative error to be computed. "
             "Timesteps below this are flagged as zero_total. (default: 1.0)"
    )
    parser.add_argument(
        "--use_ground_truth", action="store_true",
        help="Check AR6 ground truth (y_test) instead of model predictions. "
             "Useful as a sanity check that sum constraints hold in real data."
    )
    args = parser.parse_args()

    # ---- Discover hierarchy from targets ----
    # We need targets before loading everything — peek at the run store
    from src.utils.run_store import RunStore
    from scripts.train_xgb import derive_splits
    store  = RunStore(args.run_id)
    data   = store.load_processed_data()
    splits = derive_splits(data)
    all_targets = splits["targets"]

    hierarchy = discover_hierarchy(all_targets)
    if not hierarchy:
        print("No hierarchical variable relationships found in this run's targets.")
        print("Sum check requires at least one parent variable with direct children.")
        sys.exit(0)

    print(f"\n  Discovered {len(hierarchy)} parent-child group(s):")
    for parent, children in hierarchy.items():
        print(f"    {parent}  ({len(children)} children)")

    # ---- Load ----
    test_data, preds, y_test, targets = load_predictions(args.run_id, hierarchy)

    if args.use_ground_truth:
        values     = y_test
        label      = "AR6 ground truth (y_test)"
        out_subdir = "sum_check_ground_truth"
    else:
        values     = preds
        label      = "model predictions"
        out_subdir = "sum_check"

    print(f"\n  Data source: {label}")
    long = build_long(test_data, values, targets, label)

    # ---- Set up output dir and tee ----
    out_dir = REPO_ROOT / "results" / "xgb" / args.run_id / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(out_dir / "report.txt")

    print(f"\n  Running sum check (threshold={args.threshold*100:.1f}%, "
          f"abs_floor={args.abs_floor})...")

    # ---- Run check for each parent variable ----
    all_timestep_dfs = []
    for parent, children in hierarchy.items():
        print(f"\n  Checking: {parent}  ({len(children)} children)")
        tdf = run_sum_check(long, parent, children, args.threshold, args.abs_floor)
        all_timestep_dfs.append(tdf)
        print(f"    Timestep rows: {len(tdf):,}")

    timestep_df = pd.concat(all_timestep_dfs, ignore_index=True)
    summary     = scenario_summary(timestep_df, args.threshold)

    print(f"\n  Total timestep rows evaluated: {len(timestep_df):,}")
    print(f"  Total scenario-regions:        {len(summary):,}")

    # ---- Reports ----
    report_overview(summary, args.threshold)
    report_by_parent(summary, args.threshold)
    report_by_category(summary)
    report_error_distribution(summary)
    report_component_contributions(timestep_df, hierarchy)
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
