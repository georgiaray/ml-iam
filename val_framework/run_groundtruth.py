"""
Ground truth validation runner for ML-IAM.

Runs all four checks against the AR6 test-set ground truth, producing
reference results used by make_val_report.py for side-by-side comparison
with model predictions.

Should be run after run_all.py (or at least after the individual prediction
checks), since make_val_report.py expects both sets of results to exist.

Checks run (all with --use_ground_truth):
  - check_plausibility   → results/xgb/<run_id>/plausibility_ground_truth/
  - sum_check            → results/xgb/<run_id>/sum_check_ground_truth/
  - regional_consistency → results/xgb/<run_id>/regional_consistency_ground_truth/
  - bounds_check         → results/xgb/<run_id>/bounds_check_ground_truth/

For plausibility, empirical bounds are still derived from the ground truth;
the check then flags whether the ground truth itself falls within those bounds.
This gives a baseline violation rate to compare predictions against.

Usage:
    python val_framework/run_groundtruth.py --run_id xgb_04
    python val_framework/run_groundtruth.py --run_id xgb_04 --no-regional
"""

import argparse
import importlib
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_check(module_name: str, description: str, argv: list[str]) -> bool:
    print(f"\n{'#'*60}")
    print(f"  Running (ground truth): {description}")
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
    parser = argparse.ArgumentParser(
        description="Run all ground truth validation checks for ML-IAM"
    )
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
        help="Tail percentile for empirical bounds in bounds check (default: 1.0)"
    )
    parser.add_argument(
        "--grouping", type=str, default=None, choices=["R5", "R6", "R10"],
        help="Restrict regional consistency check to one grouping (default: check all)"
    )
    parser.add_argument(
        "--no_empirical", action="store_true",
        help="Disable empirical bounds in the ground truth bounds check"
    )
    parser.add_argument(
        "--by_category", action="store_true",
        help="Break down plausibility violations by scenario category"
    )
    parser.add_argument("--no-plausibility", action="store_true", help="Skip plausibility check")
    parser.add_argument("--no-sum-check",    action="store_true", help="Skip sum check")
    parser.add_argument("--no-regional",     action="store_true", help="Skip regional consistency check")
    parser.add_argument("--no-bounds",       action="store_true", help="Skip bounds check")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  ML-IAM GROUND TRUTH VALIDATION")
    print(f"  Run ID : {args.run_id}")
    print(f"{'='*60}")

    results = {}

    # --- Ground truth plausibility check ---
    if not args.no_plausibility:
        argv = [
            "--run_id",    args.run_id,
            "--percentile", str(args.percentile),
            "--use_ground_truth",
        ]
        if args.by_category:
            argv.append("--by_category")
        results["check_plausibility (GT)"] = run_check(
            "check_plausibility", "Growth rate plausibility check", argv
        )

    # --- Ground truth sum check ---
    if not args.no_sum_check:
        argv = [
            "--run_id",    args.run_id,
            "--threshold", str(args.threshold),
            "--abs_floor", str(args.abs_floor),
            "--use_ground_truth",
        ]
        results["sum_check (GT)"] = run_check("sum_check", "Hierarchy sum check", argv)

    # --- Ground truth regional consistency ---
    if not args.no_regional:
        argv = [
            "--run_id",    args.run_id,
            "--threshold", str(args.threshold),
            "--abs_floor", str(args.abs_floor),
            "--use_ground_truth",
        ]
        if args.grouping:
            argv += ["--grouping", args.grouping]
        results["regional_consistency (GT)"] = run_check(
            "regional_consistency", "Regional consistency check", argv
        )

    # --- Ground truth bounds check ---
    if not args.no_bounds:
        argv = ["--run_id", args.run_id, "--percentile", str(args.percentile),
                "--use_ground_truth"]
        if args.no_empirical:
            argv.append("--no_empirical")
        results["bounds_check (GT)"] = run_check("bounds_check", "Physical bounds check", argv)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  GROUND TRUTH VALIDATION SUMMARY")
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
