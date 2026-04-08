"""
Validation framework runner for ML-IAM predictions.

Runs all validation checks against a trained model's test-set predictions
and saves reports under results/xgb/<run_id>/<check_name>/.

Usage:
    python val_framework/run_all.py --run_id xgb_03
    python val_framework/run_all.py --run_id xgb_03 --percentile 5
    python val_framework/run_all.py --run_id xgb_03 --no-sum-check

Available flags:
    --run_id          Run ID to validate (required)
    --percentile      Tail percentile for plausibility bounds (default: 1)
    --threshold       Sum check max relative error per timestep (default: 0.012)
    --by_category     Break down plausibility violations by scenario category
    --no-plausibility Skip the plausibility check
    --no-sum-check    Skip the sum check
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
    ("check_plausibility", "Growth rate plausibility check"),
    ("sum_check",          "Secondary Energy|Electricity sum check"),
]


def run_check(module_name: str, description: str, argv: list[str]) -> bool:
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
        # Re-import to re-run (module may have been cached)
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
    parser.add_argument("--run_id", required=True, help="Run ID, e.g. xgb_03")
    parser.add_argument(
        "--percentile", type=float, default=1.0,
        help="Tail percentile for plausibility bounds (default: 1.0)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.012,
        help="Sum check max relative error per timestep (default: 0.012)"
    )
    parser.add_argument(
        "--by_category", action="store_true",
        help="Break down plausibility violations by scenario category"
    )
    parser.add_argument("--no-plausibility", action="store_true", help="Skip plausibility check")
    parser.add_argument("--no-sum-check", action="store_true", help="Skip sum check")
    args = parser.parse_args()

    skip = set()
    if args.no_plausibility:
        skip.add("check_plausibility")
    if args.no_sum_check:
        skip.add("sum_check")

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

    # --- Sum check ---
    if "sum_check" not in skip:
        argv = ["--run_id", args.run_id, "--threshold", str(args.threshold)]
        results["sum_check"] = run_check(
            "sum_check", "Secondary Energy|Electricity sum check", argv
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