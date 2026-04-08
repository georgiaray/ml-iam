"""
Validation report generator for ML-IAM predictions.

Reads the CSV outputs produced by run_all.py and generates a single Markdown
report with summary tables and figures, including side-by-side comparisons
between model predictions and AR6 ground truth wherever ground truth results
are available.

Must be run after run_all.py (or at least after the individual checks whose
results you want included). Missing check outputs are silently skipped with
a note in the report.

Usage:
    python val_framework/make_val_report.py --run_id xgb_04
    python val_framework/make_val_report.py --run_id xgb_04 --title "XGB run 04 — full variable set"

Output:
    val_framework/reports/<run_id>/report.md
    val_framework/reports/<run_id>/figures/*.png
"""

import argparse
import sys
from datetime import datetime
from typing import Optional
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO_ROOT  = Path(__file__).resolve().parent.parent
REPORTS_DIR = Path(__file__).resolve().parent / "reports"

# Consistent colour palette: predictions = blue, ground truth = orange
C_PRED = "#2c7bb6"
C_GT   = "#d7191c"
C_GRID = "#e5e5e5"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(results_base: Path, check: str, filename: str) -> Optional[pd.DataFrame]:
    path = results_base / check / filename
    if path.exists():
        return pd.read_csv(path)
    return None


def md_table(df: pd.DataFrame, fmt: Optional[dict] = None) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table.

    Pipes inside cell values are escaped so that variable names like
    'Primary Energy|Coal' do not break the table structure.
    """
    def _escape(s: str) -> str:
        return s.replace("|", "\\|")

    fmt = fmt or {}
    cols = df.columns.tolist()
    header = "| " + " | ".join(_escape(str(c)) for c in cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows   = []
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            val = row[col]
            if col in fmt:
                cells.append(_escape(fmt[col].format(val)))
            elif isinstance(val, float):
                cells.append(_escape(f"{val:.4f}"))
            else:
                cells.append(_escape(str(val)))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def save_fig(fig: plt.Figure, fig_dir: Path, name: str) -> str:
    """Save figure and return relative markdown path."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"figures/{name}.png"


GT_MISSING = "⚠ run `run_groundtruth.py`"


def _gt_missing_note(check_name: str) -> str:
    return (
        f"> **Ground truth results not found** for `{check_name}`. "
        f"Run `python val_framework/run_groundtruth.py --run_id <run_id>` "
        f"to generate reference results for comparison."
    )


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def section_overview(
    results_base: Path,
    checks_run: list,
) -> str:
    rows = []

    # Sum check
    pred_sc = load(results_base, "sum_check", "scenario_summary.csv")
    gt_sc   = load(results_base, "sum_check_ground_truth", "scenario_summary.csv")
    if pred_sc is not None:
        pr = 100 * pred_sc["passed"].mean()
        me = pred_sc["mean_error_pct"].mean()
        gt_pr = f'{100 * gt_sc["passed"].mean():.1f}%' if gt_sc is not None else GT_MISSING
        gt_me = f'{gt_sc["mean_error_pct"].mean():.3f}%' if gt_sc is not None else GT_MISSING
        rows.append({
            "Check": "Hierarchy Sum Check",
            "Metric": "Scenario-region pass rate",
            "Predictions": f"{pr:.1f}%",
            "Ground Truth": gt_pr,
        })
        rows.append({
            "Check": "",
            "Metric": "Mean relative error",
            "Predictions": f"{me:.3f}%",
            "Ground Truth": gt_me,
        })

    # Growth rate
    viol    = load(results_base, "plausibility",              "growth_rate_violations.csv")
    gt_viol = load(results_base, "plausibility_ground_truth", "growth_rate_violations.csv")
    if viol is not None:
        vr    = 100 * viol["violation"].mean()
        gt_vr = f'{100 * gt_viol["violation"].mean():.1f}%' if gt_viol is not None else GT_MISSING
        rows.append({
            "Check": "Growth Rate Plausibility",
            "Metric": "Timestep violation rate",
            "Predictions": f"{vr:.1f}%",
            "Ground Truth": gt_vr,
        })

    # Regional consistency
    pred_rc = load(results_base, "regional_consistency",              "scenario_summary.csv")
    gt_rc   = load(results_base, "regional_consistency_ground_truth", "scenario_summary.csv")
    if pred_rc is not None:
        pr    = 100 * pred_rc["passed"].mean()
        gt_pr = f'{100 * gt_rc["passed"].mean():.1f}%' if gt_rc is not None else GT_MISSING
        rows.append({
            "Check": "Regional Consistency",
            "Metric": "Scenario × variable pass rate",
            "Predictions": f"{pr:.1f}%",
            "Ground Truth": gt_pr,
        })

    # Bounds
    pred_bc = load(results_base, "bounds_check",              "scenario_summary.csv")
    gt_bc   = load(results_base, "bounds_check_ground_truth", "scenario_summary.csv")
    if pred_bc is not None:
        vr    = 100 * pred_bc["n_violations"].sum() / (pred_bc["n_timesteps"].sum() or 1)
        gt_vr = (
            f'{100 * gt_bc["n_violations"].sum() / (gt_bc["n_timesteps"].sum() or 1):.2f}%'
            if gt_bc is not None else GT_MISSING
        )
        rows.append({
            "Check": "Physical Bounds Check",
            "Metric": "Timestep violation rate",
            "Predictions": f"{vr:.2f}%",
            "Ground Truth": gt_vr,
        })

    if not rows:
        return "_No check results found._"

    df = pd.DataFrame(rows)
    return md_table(df, fmt={c: "{}" for c in df.columns})


# ---------------------------------------------------------------------------

def section_sum_check(results_base: Path, fig_dir: Path) -> tuple[str, list]:
    pred_sc = load(results_base, "sum_check", "scenario_summary.csv")
    pred_te = load(results_base, "sum_check", "timestep_errors.csv")
    gt_sc   = load(results_base, "sum_check_ground_truth", "scenario_summary.csv")
    gt_te   = load(results_base, "sum_check_ground_truth", "timestep_errors.csv")

    if pred_sc is None:
        return "_Sum check results not found. Run `sum_check.py` first._", []

    if gt_sc is None:
        blocks = [_gt_missing_note("sum_check")]
    else:
        blocks = []
    has_parent = "parent_variable" in pred_sc.columns
    figures    = []

    # --- Pass rate table ---
    if has_parent:
        tbl_data = (
            pred_sc.groupby("parent_variable")
            .agg(
                n_scenario_regions=("passed", "count"),
                pass_rate_pct=("passed", lambda x: 100 * x.mean()),
                mean_error_pct=("mean_error_pct", "mean"),
                max_error_pct=("max_error_pct", "max"),
            )
            .reset_index()
            .rename(columns={
                "parent_variable":    "Parent Variable",
                "n_scenario_regions": "Scenario-regions",
                "pass_rate_pct":      "Pass rate (%)",
                "mean_error_pct":     "Mean error (%)",
                "max_error_pct":      "Max error (%)",
            })
        )
        blocks.append("### Pass Rates by Parent Variable\n\n" + md_table(tbl_data))
    else:
        pr  = 100 * pred_sc["passed"].mean()
        me  = pred_sc["mean_error_pct"].mean()
        mx  = pred_sc["max_error_pct"].max()
        tbl = pd.DataFrame([{
            "Source": "Predictions",
            "Scenario-regions": len(pred_sc),
            "Pass rate (%)": f"{pr:.1f}",
            "Mean error (%)": f"{me:.4f}",
            "Max error (%)": f"{mx:.4f}",
        }])
        if gt_sc is not None:
            tbl = pd.concat([tbl, pd.DataFrame([{
                "Source": "Ground truth",
                "Scenario-regions": len(gt_sc),
                "Pass rate (%)": f'{100 * gt_sc["passed"].mean():.1f}',
                "Mean error (%)": f'{gt_sc["mean_error_pct"].mean():.4f}',
                "Max error (%)": f'{gt_sc["max_error_pct"].max():.4f}',
            }])], ignore_index=True)
        blocks.append("### Pass Rates\n\n" + md_table(tbl, fmt={c: "{}" for c in tbl.columns}))

    # --- Figure 1: error distribution ---
    if pred_te is not None:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        bins = np.linspace(0, min(pred_te["abs_error"].quantile(0.99) * 1.1, 1.0), 60)
        ax.hist(pred_te["abs_error"].clip(upper=bins[-1]), bins=bins,
                color=C_PRED, alpha=0.7, label="Predictions", density=True)
        if gt_te is not None:
            ax.hist(gt_te["abs_error"].clip(upper=bins[-1]), bins=bins,
                    color=C_GT, alpha=0.6, label="Ground truth", density=True)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        style_ax(ax, title="Sum Check — Error Distribution",
                 xlabel="Relative error  |parent − sum_children| / |parent|",
                 ylabel="Density")
        ax.legend(fontsize=8)
        fig.tight_layout()
        rel = save_fig(fig, fig_dir, "sum_check_error_dist")
        figures.append(rel)
        blocks.append(f"### Error Distribution\n\n![Sum check error distribution]({rel})")

    # --- Figure 2: mean error by year ---
    if pred_te is not None and "Year" in pred_te.columns:
        group_col = "parent_variable" if "parent_variable" in pred_te.columns else None

        fig, ax = plt.subplots(figsize=(7, 3.5))

        def _plot_by_year(te, label, color, linestyle="-"):
            by_year = te.groupby("Year")["abs_error"].mean() * 100
            ax.plot(by_year.index, by_year.values, marker="o", markersize=3,
                    color=color, linestyle=linestyle, linewidth=1.5, label=label)

        # gt_te may come from an older run that pre-dates the parent_variable column
        gt_group_col = group_col if (gt_te is not None and group_col in gt_te.columns) else None

        if group_col:
            for parent in pred_te[group_col].unique():
                sub = pred_te[pred_te[group_col] == parent]
                short = parent.split("|")[-1]
                _plot_by_year(sub, f"Pred: {short}", C_PRED)
            if gt_te is not None:
                if gt_group_col:
                    for parent in gt_te[gt_group_col].dropna().unique():
                        sub = gt_te[gt_te[gt_group_col] == parent]
                        short = parent.split("|")[-1]
                        _plot_by_year(sub, f"GT: {short}", C_GT, linestyle="--")
                else:
                    _plot_by_year(gt_te, "Ground truth", C_GT, linestyle="--")
        else:
            _plot_by_year(pred_te, "Predictions", C_PRED)
            if gt_te is not None:
                _plot_by_year(gt_te, "Ground truth", C_GT, linestyle="--")

        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        style_ax(ax, title="Sum Check — Mean Error by Year",
                 xlabel="Year", ylabel="Mean relative error (%)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        rel = save_fig(fig, fig_dir, "sum_check_error_by_year")
        figures.append(rel)
        blocks.append(f"### Mean Error by Year\n\n![Sum check error by year]({rel})")

    # --- GT comparison table ---
    pcts = [50, 75, 90, 95, 99]
    if gt_sc is not None:
        tbl = pd.DataFrame({
            "Percentile": [f"p{p}" for p in pcts],
            "Predictions (%)": [f"{np.percentile(pred_sc['mean_error_pct'].dropna(), p):.4f}" for p in pcts],
            "Ground truth (%)": [f"{np.percentile(gt_sc['mean_error_pct'].dropna(), p):.4f}" for p in pcts],
        })
        blocks.append("### Error Percentile Comparison — Predictions vs Ground Truth\n\n"
                      + md_table(tbl, fmt={c: "{}" for c in tbl.columns}))

    return "\n\n".join(blocks), figures


# ---------------------------------------------------------------------------

def section_plausibility(results_base: Path, fig_dir: Path) -> tuple[str, list]:
    viol    = load(results_base, "plausibility",              "growth_rate_violations.csv")
    bounds  = load(results_base, "plausibility",              "empirical_bounds.csv")
    gt_viol = load(results_base, "plausibility_ground_truth", "growth_rate_violations.csv")

    if viol is None:
        return "_Growth rate plausibility results not found. Run `check_plausibility.py` first._", []

    figures = []
    blocks  = []

    if gt_viol is None:
        blocks.append(_gt_missing_note("check_plausibility"))

    # --- Summary stats ---
    total  = len(viol)
    n_viol = viol["violation"].sum()
    vr     = 100 * n_viol / total if total else 0.0
    def _median_severity(df: pd.DataFrame) -> float:
        """Median severity of violation rows, with infinities stripped."""
        sev = df.loc[df["violation"], "severity"].replace([np.inf, -np.inf], np.nan).dropna()
        return float(sev.median()) if len(sev) else 0.0

    summary_lines = (
        f"**Total timesteps evaluated:** {total:,}  \n"
        f"**Violations:** {int(n_viol):,} ({vr:.2f}%)  \n"
        f"**Median severity** (violations only): "
        f"{_median_severity(viol):.3f} bound-widths"
    )
    if gt_viol is not None:
        gt_total  = len(gt_viol)
        gt_n_viol = gt_viol["violation"].sum()
        gt_vr     = 100 * gt_n_viol / gt_total if gt_total else 0.0
        summary_lines += (
            f"  \n\n**Ground truth — violation rate:** {gt_vr:.2f}%  \n"
            f"**Ground truth — median severity:** "
            f"{_median_severity(gt_viol):.3f} bound-widths  \n"
            f"_({vr - gt_vr:+.2f}pp difference: predictions vs ground truth)_"
        )
    blocks.append(summary_lines)

    # --- Figure: violation rate by variable, predictions vs GT ---
    def _viol_by_var(df):
        return (
            df.groupby("Variable")
            .agg(total=("violation", "count"), violations=("violation", "sum"))
            .reset_index()
            .assign(**{"rate_%": lambda d: 100 * d["violations"] / d["total"]})
        )

    by_var    = _viol_by_var(viol).sort_values("rate_%", ascending=True)
    gt_by_var = _viol_by_var(gt_viol) if gt_viol is not None else None

    fig, ax = plt.subplots(figsize=(7, max(3, len(by_var) * 0.55)))
    y_pos = np.arange(len(by_var))
    ax.barh(y_pos, by_var["rate_%"], height=0.4 if gt_by_var is not None else 0.6,
            color=C_PRED, alpha=0.8, label="Predictions")
    if gt_by_var is not None:
        gt_rates = gt_by_var.set_index("Variable").reindex(by_var["Variable"])["rate_%"].fillna(0)
        ax.barh(y_pos + 0.4, gt_rates.values, height=0.4,
                color=C_GT, alpha=0.7, label="Ground truth")
    ax.set_yticks(y_pos + (0.2 if gt_by_var is not None else 0))
    ax.set_yticklabels(by_var["Variable"], fontsize=8)
    ax.bar_label(ax.containers[0], fmt="%.1f%%", fontsize=6, padding=2)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    if gt_by_var is not None:
        ax.legend(fontsize=8)
    style_ax(ax, title="Growth Rate Violations by Variable",
             xlabel="Violation rate (%)", ylabel="")
    ax.set_xlim(0, by_var["rate_%"].max() * 1.2)
    fig.tight_layout()
    rel = save_fig(fig, fig_dir, "plausibility_violations_by_variable")
    figures.append(rel)
    blocks.append(f"### Violation Rate by Variable\n\n![Plausibility violations by variable]({rel})")

    # --- Bounds table ---
    if bounds is not None:
        tbl = bounds.copy()
        tbl.columns = ["Variable", "Lower bound", "Upper bound"]
        blocks.append("### Empirical Bounds Used (AR6 test-set percentiles)\n\n" + md_table(tbl))

    # --- Figure: severity distribution (violations only) ---
    sev = viol.loc[viol["violation"], "severity"].dropna()
    if len(sev) > 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(sev.clip(upper=sev.quantile(0.99)), bins=50, color=C_PRED, alpha=0.8)
        style_ax(ax, title="Growth Rate — Violation Severity Distribution",
                 xlabel="Severity (bound-widths outside range)", ylabel="Count")
        fig.tight_layout()
        rel = save_fig(fig, fig_dir, "plausibility_severity_dist")
        figures.append(rel)
        blocks.append(f"### Severity Distribution\n\n"
                      f"_Severity = how many bound-widths the growth rate exceeds the limit._\n\n"
                      f"![Plausibility severity]({rel})")

    # --- Violation rate by scenario category ---
    if "Scenario_Category" in viol.columns:
        cat = (
            viol.groupby("Scenario_Category")
            .agg(total=("violation", "count"), violations=("violation", "sum"))
            .reset_index()
        )
        cat["rate_%"] = (100 * cat["violations"] / cat["total"]).round(2)
        cat = cat.sort_values("rate_%", ascending=False)
        cat.columns = ["Category", "Timesteps", "Violations", "Violation rate (%)"]
        blocks.append("### Violation Rate by Scenario Category\n\n" + md_table(cat))

    return "\n\n".join(blocks), figures


# ---------------------------------------------------------------------------

def section_regional(results_base: Path, fig_dir: Path) -> tuple[str, list]:
    pred_sc = load(results_base, "regional_consistency", "scenario_summary.csv")
    gt_sc   = load(results_base, "regional_consistency_ground_truth", "scenario_summary.csv")

    if pred_sc is None:
        return ("_Regional consistency results not found. Run `regional_consistency.py` first, "
                "or skip if your run has no multi-region scenarios._"), []

    figures = []
    blocks  = []

    if gt_sc is None:
        blocks.append(_gt_missing_note("regional_consistency"))

    # --- By grouping table ---
    def _grouping_table(sc):
        return (
            sc.groupby("grouping")
            .agg(
                total=("passed", "count"),
                passed=("passed", "sum"),
                mean_error_pct=("mean_error_pct", "mean"),
                max_error_pct=("max_error_pct", "max"),
            )
            .reset_index()
            .assign(**{"pass_rate_%": lambda d: 100 * d["passed"] / d["total"]})
        )

    pred_grp = _grouping_table(pred_sc)
    tbl_data = pred_grp[["grouping", "total", "passed", "pass_rate_%",
                          "mean_error_pct", "max_error_pct"]].copy()
    tbl_data.columns = ["Grouping", "Total", "Passed", "Pass rate (%)",
                         "Mean error (%)", "Max error (%)"]
    blocks.append("### Pass Rates by Regional Grouping — Predictions\n\n"
                  + md_table(tbl_data))

    if gt_sc is not None:
        gt_grp = _grouping_table(gt_sc)
        tbl_gt = gt_grp[["grouping", "total", "passed", "pass_rate_%",
                          "mean_error_pct", "max_error_pct"]].copy()
        tbl_gt.columns = tbl_data.columns
        blocks.append("### Pass Rates by Regional Grouping — Ground Truth\n\n"
                      + md_table(tbl_gt))

    # --- Figure: pass rate bar chart by grouping & variable ---
    if "Variable" in pred_sc.columns:
        var_grp = (
            pred_sc.groupby(["grouping", "Variable"])
            .agg(total=("passed", "count"), passed=("passed", "sum"))
            .reset_index()
        )
        var_grp["pass_rate_%"] = 100 * var_grp["passed"] / var_grp["total"]
        groupings = var_grp["grouping"].unique()

        fig, axes = plt.subplots(
            1, len(groupings),
            figsize=(5 * len(groupings), max(3.5, len(var_grp["Variable"].unique()) * 0.4)),
            sharey=True,
        )
        if len(groupings) == 1:
            axes = [axes]

        for ax, g in zip(axes, sorted(groupings)):
            sub = var_grp[var_grp["grouping"] == g].sort_values("pass_rate_%")
            ax.barh(sub["Variable"], sub["pass_rate_%"], color=C_PRED, alpha=0.8, label="Predictions")
            if gt_sc is not None and "Variable" in gt_sc.columns:
                gt_sub = (
                    gt_sc[gt_sc["grouping"] == g]
                    .groupby("Variable")
                    .agg(total=("passed", "count"), passed=("passed", "sum"))
                    .reset_index()
                )
                gt_sub["pass_rate_%"] = 100 * gt_sub["passed"] / gt_sub["total"]
                gt_sub = gt_sub.set_index("Variable").reindex(sub["Variable"]).reset_index()
                ax.barh(
                    np.arange(len(sub)) + 0.3,
                    gt_sub["pass_rate_%"].fillna(0),
                    height=0.3, color=C_GT, alpha=0.7, label="Ground truth",
                )
            ax.xaxis.set_major_formatter(mticker.PercentFormatter())
            style_ax(ax, title=f"{g}", xlabel="Pass rate (%)")
            ax.set_xlim(0, 105)
            if ax == axes[-1]:
                ax.legend(fontsize=7)

        fig.suptitle("Regional Consistency — Pass Rate by Variable", fontsize=11, fontweight="bold")
        fig.tight_layout()
        rel = save_fig(fig, fig_dir, "regional_consistency_by_variable")
        figures.append(rel)
        blocks.append(f"### Pass Rate by Variable\n\n![Regional consistency by variable]({rel})")

    return "\n\n".join(blocks), figures


# ---------------------------------------------------------------------------

def section_bounds(results_base: Path, fig_dir: Path) -> tuple[str, list]:
    pred_sc  = load(results_base, "bounds_check", "scenario_summary.csv")
    pred_viol = load(results_base, "bounds_check", "violations.csv")
    bounds   = load(results_base, "bounds_check", "bounds_used.csv")
    gt_sc    = load(results_base, "bounds_check_ground_truth", "scenario_summary.csv")

    if pred_sc is None:
        return "_Bounds check results not found. Run `bounds_check.py` first._", []

    figures = []
    blocks  = []

    if gt_sc is None:
        blocks.append(_gt_missing_note("bounds_check"))

    # --- Overview stats ---
    total = pred_sc["n_timesteps"].sum()
    n_viol = pred_sc["n_violations"].sum()
    vr = 100 * n_viol / total if total else 0.0
    n_clean = pred_sc["passed"].sum()
    blocks.append(
        f"**Timesteps checked:** {total:,}  \n"
        f"**Violations:** {int(n_viol):,} ({vr:.3f}%)  \n"
        f"**Fully clean scenario-regions:** {int(n_clean):,} / {len(pred_sc):,}"
    )

    # --- Bounds table ---
    if bounds is not None:
        tbl = bounds[["Variable", "lower_bound", "upper_bound"]].copy()
        tbl.columns = ["Variable", "Lower bound", "Upper bound"]
        tbl["Lower bound"] = tbl["Lower bound"].apply(
            lambda x: f"{x:.4g}" if pd.notna(x) else "—"
        )
        tbl["Upper bound"] = tbl["Upper bound"].apply(
            lambda x: f"{x:.4g}" if pd.notna(x) else "—"
        )
        blocks.append("### Bounds Applied\n\n" + md_table(tbl, fmt={c: "{}" for c in tbl.columns}))

    # --- Figure: violation counts by variable (predictions) ---
    by_var = (
        pred_sc.groupby("Variable")
        .agg(
            n_violations=("n_violations", "sum"),
            n_timesteps=("n_timesteps", "sum"),
            n_below=("n_below_lower", "sum"),
            n_above=("n_above_upper", "sum"),
        )
        .reset_index()
    )
    by_var["rate_%"] = 100 * by_var["n_violations"] / by_var["n_timesteps"]
    by_var = by_var.sort_values("n_violations", ascending=True)

    if by_var["n_violations"].sum() > 0:
        fig, ax = plt.subplots(figsize=(7, max(3, len(by_var) * 0.45)))
        ax.barh(by_var["Variable"], by_var["n_below"],
                color=C_PRED, alpha=0.8, label="Below lower bound")
        ax.barh(by_var["Variable"], by_var["n_above"],
                left=by_var["n_below"],
                color=C_GT, alpha=0.7, label="Above upper bound")
        style_ax(ax, title="Bounds Violations by Variable",
                 xlabel="Number of violating timesteps", ylabel="")
        ax.legend(fontsize=8)
        fig.tight_layout()
        rel = save_fig(fig, fig_dir, "bounds_violations_by_variable")
        figures.append(rel)
        blocks.append(f"### Violations by Variable\n\n![Bounds violations by variable]({rel})")
    else:
        blocks.append("### Violations by Variable\n\n✓ No violations detected across any variable.")

    # --- GT comparison ---
    if gt_sc is not None:
        gt_total = gt_sc["n_timesteps"].sum()
        gt_viol  = gt_sc["n_violations"].sum()
        gt_vr    = 100 * gt_viol / gt_total if gt_total else 0.0
        diff_pp  = vr - gt_vr
        sign     = "+" if diff_pp >= 0 else ""
        tbl = pd.DataFrame([
            {"Source": "Predictions", "Timesteps": f"{total:,}",
             "Violations": f"{int(n_viol):,}", "Violation rate": f"{vr:.3f}%"},
            {"Source": "Ground truth", "Timesteps": f"{gt_total:,}",
             "Violations": f"{int(gt_viol):,}", "Violation rate": f"{gt_vr:.3f}%"},
        ])
        blocks.append(
            f"### Predictions vs Ground Truth\n\n"
            + md_table(tbl, fmt={c: "{}" for c in tbl.columns})
            + f"\n\n_Predictions show {sign}{diff_pp:.3f} pp more violations than ground truth._"
        )

        # Paired bar: violation rate by variable, predictions vs GT
        if "Variable" in gt_sc.columns and by_var["n_violations"].sum() > 0:
            gt_by_var = (
                gt_sc.groupby("Variable")
                .agg(n_violations=("n_violations", "sum"), n_timesteps=("n_timesteps", "sum"))
                .reset_index()
            )
            gt_by_var["rate_%"] = 100 * gt_by_var["n_violations"] / gt_by_var["n_timesteps"]
            merged_var = by_var[["Variable", "rate_%"]].merge(
                gt_by_var[["Variable", "rate_%"]].rename(columns={"rate_%": "gt_rate_%"}),
                on="Variable", how="left",
            ).sort_values("rate_%", ascending=True)

            fig, ax = plt.subplots(figsize=(7, max(3, len(merged_var) * 0.45)))
            y = np.arange(len(merged_var))
            ax.barh(y, merged_var["rate_%"], height=0.4, color=C_PRED, alpha=0.8, label="Predictions")
            ax.barh(y + 0.4, merged_var["gt_rate_%"].fillna(0),
                    height=0.4, color=C_GT, alpha=0.7, label="Ground truth")
            ax.set_yticks(y + 0.2)
            ax.set_yticklabels(merged_var["Variable"], fontsize=8)
            ax.xaxis.set_major_formatter(mticker.PercentFormatter())
            style_ax(ax, title="Bounds Violation Rate by Variable — Predictions vs Ground Truth",
                     xlabel="Violation rate (%)", ylabel="")
            ax.legend(fontsize=8)
            fig.tight_layout()
            rel = save_fig(fig, fig_dir, "bounds_violation_rate_pred_vs_gt")
            figures.append(rel)
            blocks.append(
                f"### Violation Rate by Variable — Predictions vs Ground Truth\n\n"
                f"![Bounds violation rate pred vs GT]({rel})"
            )

    return "\n\n".join(blocks), figures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a validation report from run_all.py outputs"
    )
    parser.add_argument("--run_id", required=True, help="Run ID, e.g. xgb_04")
    parser.add_argument(
        "--title", type=str, default=None,
        help="Optional report title (default: 'Validation Report: <run_id>')"
    )
    args = parser.parse_args()

    results_base = REPO_ROOT / "results" / "xgb" / args.run_id
    if not results_base.exists():
        print(f"ERROR: No results found at {results_base}")
        print("Run run_all.py first.")
        sys.exit(1)

    report_dir = REPORTS_DIR / args.run_id
    fig_dir    = report_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)

    title = args.title or f"Validation Report: {args.run_id}"
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"\n{'='*60}")
    print(f"  Generating validation report for run: {args.run_id}")
    print(f"  Output: {report_dir / 'report.md'}")
    print(f"{'='*60}\n")

    # Detect which checks have results
    checks_run = []
    for check in ["sum_check", "plausibility", "regional_consistency", "bounds_check"]:
        if (results_base / check).exists():
            checks_run.append(check)
            print(f"  Found: {check}/")
        else:
            print(f"  Missing (will be skipped): {check}/")

    # --- Build sections ---
    print("\n  Building sections...")

    overview = section_overview(results_base, checks_run)

    print("  Generating sum check section...")
    sc_body, sc_figs = section_sum_check(results_base, fig_dir)

    print("  Generating plausibility section...")
    pl_body, pl_figs = section_plausibility(results_base, fig_dir)

    print("  Generating regional consistency section...")
    rc_body, rc_figs = section_regional(results_base, fig_dir)

    print("  Generating bounds check section...")
    bc_body, bc_figs = section_bounds(results_base, fig_dir)

    all_figs = sc_figs + pl_figs + rc_figs + bc_figs
    print(f"  Figures generated: {len(all_figs)}")

    # --- Assemble report ---
    report = f"""# {title}

**Run ID:** `{args.run_id}`
**Generated:** {now}
**Results path:** `results/xgb/{args.run_id}/`

---

## Overview

{overview}

---

## 1. Hierarchy Sum Check

_Checks that predicted parent variables equal the sum of their direct children
at every timestep. Predictions are **expected to fail** this check — the failure
rate quantifies how much the model violates the sum constraint. Compare to the
ground truth pass rate to understand baseline data consistency._

{sc_body}

---

## 2. Growth Rate Plausibility

_For each predicted trajectory, checks that the 5-year period-on-period growth rate
falls within the 1st–99th percentile range observed in the AR6 test-set ground truth.
Empirical bounds are derived per variable._

{pl_body}

---

## 3. Regional Consistency

_Checks that predicted World values equal the sum of predicted subregion values
(R5 / R6 / R10 groupings). Only checked for scenarios where a complete grouping
is present. Predictions are **expected to fail** if the model predicts regions
independently of World._

{rc_body}

---

## 4. Physical Bounds Check

_Checks predicted values against hard physical lower bounds (energy variables ≥ 0)
and empirical per-variable bounds derived from the AR6 test-set ground truth._

{bc_body}
"""

    out_path = report_dir / "report.md"
    out_path.write_text(report)

    print(f"\n{'='*60}")
    print(f"  Report written to: {out_path}")
    if all_figs:
        print(f"  Figures saved to:  {fig_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
