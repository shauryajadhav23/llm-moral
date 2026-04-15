"""
Export random slope results to CSV and create visualization.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from statsmodels.formula.api import mixedlm

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
    """Prepare data for mixed-effects regression."""
    df = df.copy()
    df["verdict_numeric"] = df["verdict"].fillna(0).astype(int)
    df["is_sequential"] = (df["path_type"] == "sequential").astype(int)
    return df


def fit_random_slope_model(df: pd.DataFrame):
    """Fit mixed-effects model."""
    formula = "verdict_numeric ~ C(path_type, Treatment('direct')) + C(terminal_wording_id)"
    model = mixedlm(
        formula,
        data=df,
        groups=df["scenario_id"],
        re_formula="~1+is_sequential",
    )
    return model.fit(), df


def extract_random_effects(result) -> pd.DataFrame:
    """Extract random intercepts and slopes."""
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
    """Compute scenario-specific sequential effects."""
    params = result.fe_params
    seq_key = None
    for key in params.index:
        if "sequential" in key.lower():
            seq_key = key
            break

    fixed_seq_effect = params[seq_key] if seq_key else np.nan

    random_effects_df = random_effects_df.copy()
    random_effects_df["fixed_sequential_effect"] = fixed_seq_effect
    random_effects_df["conditional_sequential_effect"] = (
        fixed_seq_effect + random_effects_df["random_slope_sequential"]
    )

    scenario_class = df.groupby("scenario_id")["independence_class"].first()
    random_effects_df = random_effects_df.merge(
        scenario_class.reset_index().rename(columns={0: "independence_class"}),
        on="scenario_id",
        how="left"
    )

    return random_effects_df


def export_results(effects_df: pd.DataFrame, outdir: Path) -> None:
    """Export results to CSV files."""
    outdir.mkdir(exist_ok=True, parents=True)

    # Full scenario effects
    effects_df.to_csv(outdir / "random_effects_by_scenario.csv", index=False)
    print(f"Exported: {outdir / 'random_effects_by_scenario.csv'}")

    # Summary by independence class
    by_class = effects_df.groupby("independence_class").agg({
        "conditional_sequential_effect": ["mean", "std", "min", "max", "count"],
        "random_slope_sequential": ["mean", "std"],
    }).round(4)
    by_class.to_csv(outdir / "sequential_effect_by_class.csv")
    print(f"Exported: {outdir / 'sequential_effect_by_class.csv'}")


def create_visualization(effects_df: pd.DataFrame, outdir: Path) -> None:
    """Create visualization of conditional effects by scenario and class."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed, skipping visualization")
        return

    outdir.mkdir(exist_ok=True, parents=True)

    # Plot 1: Conditional sequential effects by scenario, colored by independence class
    fig, ax = plt.subplots(figsize=(14, 7))

    class_colors = {"commutative": "green", "partial": "orange", "entangled": "red"}
    effects_df_sorted = effects_df.sort_values("conditional_sequential_effect")

    for independence_class in ["commutative", "partial", "entangled"]:
        subset = effects_df_sorted[effects_df_sorted["independence_class"] == independence_class]
        ax.barh(
            subset["scenario_id"],
            subset["conditional_sequential_effect"],
            label=independence_class,
            color=class_colors[independence_class],
            alpha=0.7,
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Conditional Sequential Effect (BLUP)")
    ax.set_title("Scenario-Specific Sequential Path Effects (with Random Slope)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / "conditional_effects_by_scenario.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {outdir / 'conditional_effects_by_scenario.png'}")
    plt.close()

    # Plot 2: Distribution by independence class
    fig, ax = plt.subplots(figsize=(10, 6))

    for independence_class in ["commutative", "partial", "entangled"]:
        subset = effects_df[effects_df["independence_class"] == independence_class]
        ax.scatter(
            [independence_class] * len(subset),
            subset["conditional_sequential_effect"],
            s=100,
            alpha=0.6,
            color=class_colors[independence_class],
            label=independence_class,
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Conditional Sequential Effect")
    ax.set_title("Distribution of Sequential Path Effects by Independence Class")
    plt.tight_layout()
    plt.savefig(outdir / "effects_distribution_by_class.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {outdir / 'effects_distribution_by_class.png'}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="outputs/runs.db")
    parser.add_argument("--outdir", default="outputs/analysis")
    args = parser.parse_args()

    with DB(args.db) as db:
        df_raw = load_run_results(db)

    if df_raw.empty:
        print("No completed runs found.")
        sys.exit(1)

    df = prepare_analysis_data(df_raw)
    result, df_used = fit_random_slope_model(df)
    random_effects_df = extract_random_effects(result)
    effects_df = compute_scenario_specific_effects(result, random_effects_df, df_used)

    outdir = Path(args.outdir)
    export_results(effects_df, outdir)
    create_visualization(effects_df, outdir)

    print("\n✓ Random slope results exported")


if __name__ == "__main__":
    main()
