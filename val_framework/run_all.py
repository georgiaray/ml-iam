"""
Validation framework runner for ML-IAM predictions.

Runs all validation checks against a trained model's test-set predictions
and saves reports under results/xgb/<run_id>/<check_name>/.

Usage:
    python val_framework/run_all.py --run_id xgb_04
    python val_framework/run_all.py --run_id xgb_04 --percentile 5
    python val_framework/run_all.py --run_id xgb_04 --no-sum-check
    python val_framework/run_all.py --run_id xgb_04 --grouping R10

Available flags:
    --run_id              Run ID to validate (required)
    --threshold           Relative error threshold for sum/regional checks (default: 0.012)
    --abs_floor           Min absolute value for relative error to be computed (default: 1.0)
    --percentile          Tail percentile for plausibility / empirical bounds (default: 1)
    --grouping            Restrict regional check to one grouping: R5, R6, or R10
    --no_empirical        Disable empirical bounds in the bounds check (physical bounds only)
    --by_category         Break down plausibility violations by scenario category
    --no-plausibility     Skip the plausibility check
    --no-sum-check        Skip the hierarchy sum check
    --no-regional         Skip the regional consistency check
    --no-bounds           Skip the bounds check
    --no-groundtruth      Skip the ground truth reference runs (run_groundtruth.py)
    --report              Generate a validation report after all checks complete
"""

import argparse
import importlib
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Registry of checks: (module_name, description)
# To add a new check, append an entry here and create the corresponding module
# in val_framework/. The module must expose a main() that accepts sys.argv.
CHECKS = [
    ("check_plausibility",   "Growth rate plausibility check"),
    ("sum_check",            "Hierarchy sum check"),
    ("regional_consistency", "Regional consistency check"),
    ("bounds_check",         "Physical bounds check"),
]


def run_check(module_name: str, description: str, argv: list) -> bool:
    """
    Import and run a check module's main() with the given argv.
    Returns True if the check completed without error, False otherwise.
    """
    print(f"\n{'#'*60}")
    print(f"  Running: {description}")
    print(f"{'#'*60}")

    original_argv = sys.argv
    try:
        sys.argv = [module_name] + argv
        mod = importlib.import_module(module_name)
        importlib.reload(mod)
        mod.main()
        return True
    except SystemExit as e:
        if e.code == 0 or e.code is None:
            return True
        print(f"\n  [ERROR] {description} exited with code {e.code}")
        return False
    except Exception:
        print(f"\n  [ERROR] {description} raised an exception:")
        traceback.print_exc()
        return False
    finally:
        sys.argv = original_argv


def main():
    parser = argparse.ArgumentParser(description="Run all ML-IAM validation checks")
    parser.add_argument("--run_id", required=True, help="Run ID, e.g. xgb_04")
    parser.add_argument(
        "--threshold", type=float, default=0.012,
        help="Max relative error per timestep for sum/regional checks (default: 0.012)"
    )
    parser.add_argument(
        "--abs_floor", type=float, default=1.0,
        help="Min absolute value for relative error to be computed (default: 1.0)"
    )
    parser.add_argument(
        "--percentile", type=float, default=1.0,
        help="Tail percentile for plausibility bounds and empirical bounds (default: 1.0)"
    )
    parser.add_argument(
        "--grouping", type=str, default=None, choices=["R5", "R6", "R10"],
        help="Restrict regional consistency check to one grouping (default: check all)"
    )
    parser.add_argument(
        "--no_empirical", action="store_true",
        help="Disable empirical bounds in the bounds check (use physical bounds only)"
    )
    parser.add_argument(
        "--by_category", action="store_true",
        help="Break down plausibility violations by scenario category"
    )
    parser.add_argument("--no-plausibility",  action="store_true", help="Skip plausibility check")
    parser.add_argument("--no-sum-check",     action="store_true", help="Skip hierarchy sum check")
    parser.add_argument("--no-regional",      action="store_true", help="Skip regional consistency check")
    parser.add_argument("--no-bounds",        action="store_true", help="Skip bounds check")
    parser.add_argument("--no-groundtruth",   action="store_true",
                        help="Skip ground truth reference runs (skips run_groundtruth.py)")
    parser.add_argument("--report",           action="store_true",
                        help="Generate a validation report after all checks complete")
    args = parser.parse_args()

    skip = set()
    if args.no_plausibility: skip.add("check_plausibility")
    if args.no_sum_check:    skip.add("sum_check")
    if args.no_regional:     skip.add("regional_consistency")
    if args.no_bounds:       skip.add("bounds_check")

    print(f"\n{'='*60}")
    print(f"  ML-IAM VALIDATION FRAMEWORK")
    print(f"  Run ID : {args.run_id}")
    print(f"{'='*60}")

    results = {}

    # --- Plausibility check ---
    if "check_plausibility" not in skip:
        argv = ["--run_id", args.run_id, "--percentile", str(args.percentile)]
        if args.by_category:
            argv.append("--by_category")
        results["check_plausibility"] = run_check(
            "check_plausibility", "Growth rate plausibility check", argv
        )

    # --- Hierarchy sum check ---
    if "sum_check" not in skip:
        argv = [
            "--run_id",    args.run_id,
            "--threshold", str(args.threshold),
            "--abs_floor", str(args.abs_floor),
        ]
        results["sum_check"] = run_check(
            "sum_check", "Hierarchy sum check", argv
        )

    # --- Regional consistency check ---
    if "regional_consistency" not in skip:
        argv = [
            "--run_id",    args.run_id,
            "--threshold", str(args.threshold),
            "--abs_floor", str(args.abs_floor),
        ]
        if args.grouping:
            argv += ["--grouping", args.grouping]
        results["regional_consistency"] = run_check(
            "regional_consistency", "Regional consistency check", argv
        )

    # --- Bounds check ---
    if "bounds_check" not in skip:
        argv = ["--run_id", args.run_id, "--percentile", str(args.percentile)]
        if args.no_empirical:
            argv.append("--no_empirical")
        results["bounds_check"] = run_check(
            "bounds_check", "Physical bounds check", argv
        )

    # --- Ground truth reference runs ---
    if not args.no_groundtruth:
        gt_argv = [
            "--run_id",    args.run_id,
            "--threshold", str(args.threshold),
            "--abs_floor", str(args.abs_floor),
            "--percentile", str(args.percentile),
        ]
        if args.no_plausibility: gt_argv.append("--no-plausibility")
        if args.no_sum_check:    gt_argv.append("--no-sum-check")
        if args.no_regional:     gt_argv.append("--no-regional")
        if args.no_bounds:       gt_argv.append("--no-bounds")
        if args.no_empirical:    gt_argv.append("--no_empirical")
        if args.by_category:     gt_argv.append("--by_category")
        if args.grouping:        gt_argv += ["--grouping", args.grouping]
        results["run_groundtruth"] = run_check(
            "run_groundtruth", "Ground truth reference runs", gt_argv
        )

    # --- Optional report ---
    if args.report:
        results["make_val_report"] = run_check(
            "make_val_report", "Validation report generation",
            ["--run_id", args.run_id]
        )

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  VALIDATION FRAMEWORK SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for name, passed in results.items():
        status = "OK" if passed else "ERROR"
        print(f"  [{status:>5}]  {name}")
        if not passed:
            all_passed = False

    if not results:
        print("  No checks were run.")

    print(f"{'='*60}\n")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
