"""
Random slope mixed-effects model for path-dependence heterogeneity.

Fits: verdict ~ path_type + terminal_wording_id + (1 + sequential | scenario_id)

This model allows the effect of sequential path type to vary by scenario,
capturing scenario-specific heterogeneity in path-dependence.

Usage:
    python analysis/random_slope.py --db outputs/runs.db

Output:
    - Population-average path_type and wording effects
    - Variance components (intercept, sequential slope)
    - By-scenario random effects (BLUPs)
    - Scenario-specific sequential effect estimates
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from statsmodels.formula.api import mixedlm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial

from src.storage import DB


def load_run_results(db: DB) -> pd.DataFrame:
    """Load completed run results, include scenario metadata."""
    sql = """
    SELECT
        r.run_id,
        r.scenario_id,
        r.path_type,
        r.terminal_wording_id,
        r.verdict,
        r.status,
        s.independence_class
    FROM runs r
    LEFT JOIN scenarios s ON r.scenario_id = s.scenario_id
    WHERE r.status IN ('complete', 'refused')
    ORDER BY r.scenario_id, r.path_type
    """
    return db.to_dataframe(sql, [])


def prepare_analysis_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for mixed-effects regression.

    - verdict: binary (1=yes, 0=no, NaN=refused treated as 0)
    - path_type: categorical (direct is baseline)
    - terminal_wording_id: categorical
    - scenario_id: grouping variable
    - is_sequential: binary indicator for random slope
    """
    # Treat refused as 0 (conservative)
    df = df.copy()
    df["verdict_numeric"] = df["verdict"].fillna(0).astype(int)

    # Create binary sequential indicator for random slope
    df["is_sequential"] = (df["path_type"] == "sequential").astype(int)

    return df


def fit_random_slope_model(df: pd.DataFrame):
    """
    Fit mixed-effects logistic model with random slope for sequential effect.

    Formula: verdict ~ C(path_type, Treatment('direct')) + C(terminal_wording_id)
    Random effects: (1 + is_sequential | scenario_id)

    Note: statsmodels.MixedLM fits on linear predictor scale (linear mixed model),
    not full binomial GLMM. For exact logit we'd need statsmodels.genmod.generalized_linear_mixed_effects,
    but that's not yet available. MixedLM on binary outcome is reasonable approximation.
    """
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.families import Binomial

    # First try: Linear mixed model on binary outcome (reasonable approximation)
    formula = "verdict_numeric ~ C(path_type, Treatment('direct')) + C(terminal_wording_id)"

    model = mixedlm(
        formula,
        data=df,
        groups=df["scenario_id"],
        re_formula="~1+is_sequential",  # Random intercept + random slope for sequential
    )
    result = model.fit()

    return result, df


def extract_random_effects(result) -> pd.DataFrame:
    """Extract random intercepts and slopes (BLUPs) for each scenario."""
    random_effects = result.random_effects

    rows = []
    for scenario_id, effects_dict in random_effects.items():
        rows.append({
            "scenario_id": scenario_id,
            "random_intercept": effects_dict.get("Group", np.nan),
            "random_slope_sequential": effects_dict.get("is_sequential", np.nan),
        })

    return pd.DataFrame(rows)


def compute_scenario_specific_effects(
    result, random_effects_df: pd.DataFrame, df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute the scenario-specific sequential effect as:
    fixed_effect_sequential + random_slope_sequential
    """
    # Extract fixed effect coefficient for sequential
    params = result.fe_params
    seq_key = None
    for key in params.index:
        if "sequential" in key.lower():
            seq_key = key
            break

    if seq_key is None:
        print("Warning: could not find sequential coefficient in fixed effects")
        fixed_seq_effect = np.nan
    else:
        fixed_seq_effect = params[seq_key]

    # Compute conditional effects (scenario-specific)
    random_effects_df = random_effects_df.copy()
    random_effects_df["fixed_sequential_effect"] = fixed_seq_effect
    random_effects_df["conditional_sequential_effect"] = (
        fixed_seq_effect + random_effects_df["random_slope_sequential"]
    )

    # Merge in independence_class
    scenario_class = df.groupby("scenario_id")["independence_class"].first()
    random_effects_df = random_effects_df.merge(
        scenario_class.reset_index().rename(columns={0: "independence_class"}),
        on="scenario_id",
        how="left"
    )

    return random_effects_df


def print_summary(result, random_effects_df: pd.DataFrame) -> None:
    """Print model summary and random effects."""
    print("\n" + "="*80)
    print("MIXED-EFFECTS MODEL: verdict ~ path_type + terminal_wording_id + (1+sequential|scenario_id)")
    print("="*80)

    print("\n--- Fixed Effects (Population Average) ---")
    print(result.summary().tables[1])

    print("\n--- Variance Components ---")
    cov_re = result.cov_re
    print(f"Random intercept variance: {cov_re.iloc[0, 0]:.6f}")
    print(f"Random slope (sequential) variance: {cov_re.iloc[1, 1]:.6f}")
    print(f"Covariance(intercept, slope): {cov_re.iloc[0, 1]:.6f}")

    print("\n--- By-Scenario Random Effects (BLUPs) ---")
    print(random_effects_df.to_string(index=False))

    print("\n--- Scenario-Specific Sequential Effects ---")
    effects_summary = random_effects_df[
        ["scenario_id", "independence_class", "fixed_sequential_effect",
         "random_slope_sequential", "conditional_sequential_effect"]
    ].copy()
    effects_summary.columns = [
        "Scenario", "Independence Class", "Fixed Effect", "Random Slope", "Conditional Effect"
    ]
    print(effects_summary.to_string(index=False))

    # Summary by independence class
    print("\n--- Mean Sequential Effect by Independence Class ---")
    by_class = random_effects_df.groupby("independence_class")[
        "conditional_sequential_effect"
    ].agg(["mean", "std", "min", "max", "count"])
    print(by_class)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random slope mixed-effects model for path-dependence heterogeneity"
    )
    parser.add_argument("--db", default="outputs/runs.db")
    args = parser.parse_args()

    with DB(args.db) as db:
        df_raw = load_run_results(db)

    if df_raw.empty:
        print("No completed runs found.")
        sys.exit(1)

    print(f"\nLoaded {len(df_raw)} completed runs")

    df = prepare_analysis_data(df_raw)

    # Fit model
    result, df_used = fit_random_slope_model(df)

    # Extract and compute effects
    random_effects_df = extract_random_effects(result)
    effects_df = compute_scenario_specific_effects(result, random_effects_df, df_used)

    # Print results
    print_summary(result, effects_df)


if __name__ == "__main__":
    main()
