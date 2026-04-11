"""
Primary statistical test: logistic mixed-effects regression on verdict.

Model (from spec §10):
    verdict ~ path_type * terminal_wording_id + (1 | scenario_id) + (1 | perm_id)

Two implementations are provided:
  1. statsmodels MixedLM — available in pip; approximates the model (linear
     mixed effects on a binary outcome, acceptable for exploration)
  2. pymer4 — wraps R's lme4 Binomial GLMM; gold standard, requires R + lme4

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
# statsmodels implementation (accessible without R)
# ---------------------------------------------------------------------------

def run_statsmodels(df: pd.DataFrame) -> None:
    """Run logistic regression + linear mixed-effects as fallback."""
    import statsmodels.formula.api as smf

    print("\n=== Logistic Regression (fixed effects only — no random effects) ===")
    logit_model = smf.logit(
        "verdict ~ C(path_type, Treatment('direct')) * C(terminal_wording_id)",
        data=df,
    ).fit(disp=False)
    print(logit_model.summary())

    print("\n=== Linear Mixed-Effects (approximation; use pymer4 for GLMM) ===")
    lme_model = smf.mixedlm(
        "verdict ~ C(path_type, Treatment('direct')) * C(terminal_wording_id)",
        data=df,
        groups=df["scenario_id"],
        re_formula="~1",
    ).fit(reml=False)
    print(lme_model.summary())

    # Interpretation helper
    print("\n--- Coefficient table (path_type fixed effects) ---")
    params = lme_model.params
    pvals = lme_model.pvalues
    path_coeffs = {k: (v, pvals[k]) for k, v in params.items() if "path_type" in k}
    if path_coeffs:
        header = f"{'Coefficient':<55} {'Estimate':>10} {'p-value':>10}"
        print(header)
        print("-" * len(header))
        for name, (est, pv) in sorted(path_coeffs.items()):
            sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
            print(f"{name:<55} {est:>10.4f} {pv:>10.4f} {sig}")
    else:
        print("No path_type coefficients found — check data.")


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
        help="Use pymer4/lme4 GLMM instead of statsmodels (requires R)",
    )
    args = parser.parse_args()

    with DB(args.db) as db:
        df = load_results(db)

    if df.empty:
        print("No completed runs found.")
        sys.exit(1)

    print(f"Loaded {len(df)} completed runs.")
    df = prepare(df)
    print_condition_summary(df)

    if args.use_pymer4:
        run_pymer4(df)
    else:
        run_statsmodels(df)


if __name__ == "__main__":
    main()
