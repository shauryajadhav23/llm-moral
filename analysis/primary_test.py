"""
Primary statistical test: logistic regression on verdict.

Model:
    verdict ~ path_type * terminal_wording_id + scenario_id
    family = Binomial (logit link)

scenario_id is included as a fixed-effect covariate (dummy-coded) to absorb
scenario-level baseline differences in verdict rate, analogous to a within-
scenario design. Coefficients on path_type are therefore interpreted as the
average within-scenario path effect.

An alternative pymer4 implementation (wraps R's lme4 GLMM) is also provided
as the gold standard if R is available.

Usage:
    python analysis/primary_test.py --db outputs/runs.db [--use-pymer4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.storage import DB


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(db: DB) -> pd.DataFrame:
    sql = """
    SELECT
        r.run_id,
        r.scenario_id,
        r.path_type,
        r.path_signature,
        r.perm_id,
        r.terminal_wording_id,
        r.replicate_idx,
        r.verdict,
        r.status,
        s.independence_class
    FROM runs r
    JOIN scenarios s ON r.scenario_id = s.scenario_id
    WHERE r.status = 'complete'
      AND r.verdict IS NOT NULL
    """
    return db.to_dataframe(sql)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["verdict"] = df["verdict"].astype(float)
    # Reference levels: direct path, W1 wording
    df["path_type"] = pd.Categorical(
        df["path_type"],
        categories=["direct", "sequential", "skipped", "alt_grouping", "length_matched"],
    )
    df["terminal_wording_id"] = pd.Categorical(
        df["terminal_wording_id"],
        categories=sorted(df["terminal_wording_id"].unique()),
    )
    return df


# ---------------------------------------------------------------------------
# Floor/ceiling filter
# ---------------------------------------------------------------------------

def filter_ambiguous_scenarios(
    df: pd.DataFrame,
    lo: float = 0.2,
    hi: float = 0.8,
) -> pd.DataFrame:
    """
    Remove scenarios where the direct path is at floor or ceiling.

    A scenario where direct gets 0% or 100% yes is already decided —
    the model always answers the same way regardless of path. Path
    dependence is undetectable there. This filter keeps only scenarios
    where the direct baseline sits in the ambiguous range [lo, hi],
    where any path effect has room to show up.

    Uses the mean verdict across all direct runs for the scenario
    (collapsed over wordings and replicates) as the filter criterion.
    """
    direct = df[df["path_type"] == "direct"]
    scenario_p_yes = direct.groupby("scenario_id")["verdict"].mean()
    ambiguous = scenario_p_yes[
        (scenario_p_yes >= lo) & (scenario_p_yes <= hi)
    ].index
    filtered = df[df["scenario_id"].isin(ambiguous)].copy()
    n_before = df["scenario_id"].nunique()
    n_after = filtered["scenario_id"].nunique()
    print(
        f"\n--- Floor/ceiling filter (direct p_yes in [{lo}, {hi}]) ---"
        f"\n  Scenarios before: {n_before}"
        f"\n  Scenarios after:  {n_after} ({n_before - n_after} removed)"
        f"\n  Rows before: {len(df)}"
        f"\n  Rows after:  {len(filtered)}"
    )
    if n_after == 0:
        print("WARNING: No scenarios passed the filter. Loosen --floor and --ceiling.")
    return filtered


# ---------------------------------------------------------------------------
# Logistic regression (primary)
# ---------------------------------------------------------------------------

def run_logistic(df: pd.DataFrame) -> None:
    """
    Run logistic regression with scenario_id as a fixed-effect covariate.

    Formula:
        verdict ~ C(path_type, Treatment('direct'))
                  * C(terminal_wording_id)
                  + C(scenario_id)

    scenario_id dummies absorb between-scenario baseline differences so that
    path_type coefficients reflect within-scenario path effects.
    Coefficients are on the log-odds scale.
    """
    model = smf.logit(
        "verdict ~ C(path_type, Treatment('direct')) * C(terminal_wording_id)"
        " + C(scenario_id)",
        data=df,
    )
    result = model.fit(method="bfgs", disp=False)

    print("\n=== Logistic Regression (primary test) ===")
    print(result.summary())

    print("\n--- Path type effects vs direct (log-odds scale) ---")
    params = result.params
    pvals = result.pvalues
    ci = result.conf_int()
    path_coeffs = {k: (params[k], pvals[k], ci.loc[k, 0], ci.loc[k, 1])
                   for k in params.index if "path_type" in k}
    if path_coeffs:
        header = f"{'Coefficient':<55} {'Log-OR':>8} {'95% CI':>18} {'p-value':>10}"
        print(header)
        print("-" * len(header))
        for name, (est, pv, lo, hi) in sorted(path_coeffs.items()):
            sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
            print(f"{name:<55} {est:>8.4f}  [{lo:>6.3f}, {hi:>6.3f}] {pv:>10.4f} {sig}")
    else:
        print("No path_type coefficients found — check data.")


# ---------------------------------------------------------------------------
# Logistic regression by wording (separate models per wording)
# ---------------------------------------------------------------------------

def run_logistic_by_wording(df: pd.DataFrame) -> None:
    """
    Run a separate logistic regression for each terminal_wording_id.

    Each model uses scenario_id as a fixed-effect covariate and drops the
    wording term since the subset is a single wording.
    """
    wordings = sorted(df["terminal_wording_id"].cat.categories.tolist())

    for wording in wordings:
        subset = df[df["terminal_wording_id"] == wording].copy()
        n_scenarios = subset["scenario_id"].nunique()
        n_obs = len(subset)

        print(f"\n{'='*70}")
        print(f"  Wording: {wording}  |  scenarios: {n_scenarios}  |  obs: {n_obs}")
        print(f"{'='*70}")

        if n_scenarios < 3:
            print(f"  Skipping — too few scenarios ({n_scenarios}) for stable estimates.")
            continue

        try:
            model = smf.logit(
                "verdict ~ C(path_type, Treatment('direct')) + C(scenario_id)",
                data=subset,
            )
            result = model.fit(method="bfgs", disp=False)
        except Exception as e:
            print(f"  Model failed: {e}")
            continue

        params = result.params
        pvals = result.pvalues
        ci = result.conf_int()

        header = f"  {'Path type':<40} {'Log-OR':>8} {'95% CI':>18} {'p-value':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for k in params.index:
            if "path_type" not in k:
                continue
            est = params[k]
            pv = pvals[k]
            lo, hi = ci.loc[k, 0], ci.loc[k, 1]
            sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
            label = k.replace("C(path_type, Treatment('direct'))[T.", "").rstrip("]")
            print(f"  {label:<40} {est:>8.4f}  [{lo:>6.3f}, {hi:>6.3f}] {pv:>10.4f} {sig}")

        print(f"\n  Raw p_yes by path_type (wording={wording}):")
        props = subset.groupby("path_type")["verdict"].agg(n="count", p_yes="mean").reset_index()
        for _, row in props.iterrows():
            print(f"    {row['path_type']:<20} {row['p_yes']:.3f}  (n={int(row['n'])})")


# ---------------------------------------------------------------------------
# pymer4 implementation (gold standard)
# ---------------------------------------------------------------------------

def run_pymer4(df: pd.DataFrame) -> None:
    """Run GLMM with Binomial family using pymer4 (wraps R lme4)."""
    try:
        from pymer4.models import Lmer
    except ImportError:
        print("pymer4 not installed. Install with: pip install pymer4 (requires R + lme4)")
        sys.exit(1)

    print("\n=== Binomial GLMM via pymer4 (lme4) ===")
    formula = (
        "verdict ~ path_type * terminal_wording_id + "
        "(1 | scenario_id) + (1 | perm_id)"
    )
    model = Lmer(formula, data=df, family="binomial")
    model.fit(
        factors={"path_type": ["direct", "sequential", "skipped", "alt_grouping", "length_matched"]},
        summarize=False,
    )
    print(model.coefs)
    print(f"\nModel AIC: {model.AIC:.2f}")
    print(f"Marginal R²: {model.marginal_r2:.4f}")
    print(f"Conditional R²: {model.conditional_r2:.4f}")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_condition_summary(df: pd.DataFrame) -> None:
    print("\n=== Verdict proportions by path_type ===")
    summary = df.groupby("path_type")["verdict"].agg(
        n="count",
        p_yes="mean",
        std="std",
    ).reset_index()
    summary["95_ci_lo"] = summary["p_yes"] - 1.96 * summary["std"] / np.sqrt(summary["n"])
    summary["95_ci_hi"] = summary["p_yes"] + 1.96 * summary["std"] / np.sqrt(summary["n"])
    print(summary.to_string(index=False))

    print("\n=== Verdict proportions by path_type × terminal_wording_id ===")
    cross = df.groupby(["path_type", "terminal_wording_id"])["verdict"].agg(
        n="count", p_yes="mean"
    ).reset_index()
    print(cross.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Primary statistical test")
    parser.add_argument("--db", default="outputs/runs.db")
    parser.add_argument(
        "--use-pymer4",
        action="store_true",
        help="Use pymer4/lme4 GLMM instead of GEE (requires R + lme4)",
    )
    parser.add_argument(
        "--ambiguous-only",
        action="store_true",
        help=(
            "Filter to scenarios where the direct path is ambiguous "
            "(p_yes between --floor and --ceiling). Removes floor/ceiling "
            "scenarios where path dependence is undetectable by definition."
        ),
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=0.2,
        help="Lower bound for ambiguous filter (default: 0.2)",
    )
    parser.add_argument(
        "--ceiling",
        type=float,
        default=0.8,
        help="Upper bound for ambiguous filter (default: 0.8)",
    )
    parser.add_argument(
        "--by-wording",
        action="store_true",
        help=(
            "Run a separate logistic regression for each terminal wording (W1, W2, W3). "
            "Directly tests path_type significance within each wording "
            "rather than relying on interaction terms."
        ),
    )
    args = parser.parse_args()

    with DB(args.db) as db:
        df = load_results(db)

    if df.empty:
        print("No completed runs found.")
        sys.exit(1)

    print(f"Loaded {len(df)} completed runs.")
    df = prepare(df)

    if args.ambiguous_only:
        df = filter_ambiguous_scenarios(df, lo=args.floor, hi=args.ceiling)
        if df.empty:
            sys.exit(1)

    print_condition_summary(df)

    if args.by_wording:
        run_logistic_by_wording(df)
    elif args.use_pymer4:
        run_pymer4(df)
    else:
        run_logistic(df)


if __name__ == "__main__":
    main()
