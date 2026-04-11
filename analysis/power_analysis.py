"""
Power analysis and sample-size justification.

Run BEFORE the full sweep (on pilot data) to determine whether 30 replicates
is sufficient to detect the expected effect size at α=0.05, power=0.8.

Usage:
    python analysis/power_analysis.py --db outputs/runs.db --pilot-scenario util-catastrophe-01

Output:
    - Within-condition variance from pilot data
    - Required n for two-proportion z-test
    - Recommendation: proceed with sweep or increase replicates
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_effectsize, zt_ind_solve_power

from src.storage import DB


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_pilot_results(db: DB, scenario_id: str | None = None) -> pd.DataFrame:
    """Load completed run results from DB, optionally filtered to one scenario."""
    sql = """
    SELECT
        scenario_id,
        path_type,
        path_signature,
        terminal_wording_id,
        perm_id,
        replicate_idx,
        verdict,
        status
    FROM runs
    WHERE status IN ('complete', 'refused')
    """
    params = []
    if scenario_id:
        sql += " AND scenario_id = ?"
        params.append(scenario_id)

    return db.to_dataframe(sql, params)


def within_condition_variance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute verdict proportion and variance per (scenario, path_type, wording) cell."""
    grp = df[df["verdict"].notna()].groupby(
        ["scenario_id", "path_type", "terminal_wording_id"]
    )
    result = grp["verdict"].agg(
        n="count",
        p_yes="mean",
    ).reset_index()
    result["variance"] = result["p_yes"] * (1 - result["p_yes"])
    result["std"] = np.sqrt(result["variance"] / result["n"])
    return result


def estimate_required_n(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Return required replicates per condition for a two-proportion z-test."""
    effect = proportion_effectsize(p1, p2)
    n = zt_ind_solve_power(
        effect_size=abs(effect),
        alpha=alpha,
        power=power,
        alternative="two-sided",
    )
    return int(np.ceil(n))


def compute_observed_power(
    df: pd.DataFrame,
    n_replicates: int = 30,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """For each pair of path_types within a scenario/wording, compute observed power."""
    rows = []
    for (scenario_id, wording_id), sub in df.groupby(["scenario_id", "terminal_wording_id"]):
        path_types = sub["path_type"].unique()
        for i, pt1 in enumerate(path_types):
            for pt2 in path_types[i + 1:]:
                p1 = sub[sub["path_type"] == pt1]["p_yes"].values
                p2 = sub[sub["path_type"] == pt2]["p_yes"].values
                if len(p1) == 0 or len(p2) == 0:
                    continue
                p1_val = float(p1[0]) if len(p1) == 1 else float(np.mean(p1))
                p2_val = float(p2[0]) if len(p2) == 1 else float(np.mean(p2))
                if p1_val == p2_val:
                    continue
                effect = proportion_effectsize(p1_val, p2_val)
                try:
                    pwr = zt_ind_solve_power(
                        effect_size=abs(effect),
                        nobs1=n_replicates,
                        alpha=alpha,
                        alternative="two-sided",
                    )
                except Exception:
                    pwr = float("nan")
                required_n = estimate_required_n(p1_val, p2_val, alpha=alpha)
                rows.append({
                    "scenario_id": scenario_id,
                    "terminal_wording_id": wording_id,
                    "path_type_A": pt1,
                    "path_type_B": pt2,
                    "p_yes_A": p1_val,
                    "p_yes_B": p2_val,
                    "effect_size_h": abs(effect),
                    "power_at_n30": pwr,
                    "required_n_80pct": required_n,
                })
    return pd.DataFrame(rows)


def refusal_rate_by_path_type(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("path_type")["status"].apply(
        lambda s: (s == "refused").mean()
    ).reset_index()
    grp.columns = ["path_type", "refusal_rate"]
    return grp


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Power analysis from pilot data")
    parser.add_argument("--db", default="outputs/runs.db")
    parser.add_argument("--pilot-scenario", default=None, help="Filter to one scenario ID")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--target-power", type=float, default=0.8)
    parser.add_argument("--n-replicates", type=int, default=30)
    args = parser.parse_args()

    with DB(args.db) as db:
        df = load_pilot_results(db, scenario_id=args.pilot_scenario)

    if df.empty:
        print("No completed runs found. Run pilot first.")
        sys.exit(1)

    print(f"\nPilot data: {len(df)} completed runs\n")

    # Within-condition variance
    variance_df = within_condition_variance(df)
    print("=== Within-condition verdict proportions ===")
    print(variance_df.to_string(index=False))

    # Refusal rates
    refusal_df = refusal_rate_by_path_type(df)
    print("\n=== Refusal rate by path type ===")
    print(refusal_df.to_string(index=False))

    # Power analysis
    power_df = compute_observed_power(
        variance_df,
        n_replicates=args.n_replicates,
        alpha=args.alpha,
    )
    if not power_df.empty:
        print("\n=== Power analysis (pairwise path-type comparisons) ===")
        print(power_df.to_string(index=False))

        min_power = power_df["power_at_n30"].min()
        max_required_n = power_df["required_n_80pct"].max()
        print(f"\nMinimum observed power at n={args.n_replicates}: {min_power:.3f}")
        print(f"Max required n for 80% power: {max_required_n}")

        if min_power < args.target_power:
            print(
                f"\nWARNING: Power below target {args.target_power} for some comparisons. "
                f"Consider increasing replicates to {max_required_n}."
            )
        else:
            print(f"\nPower adequate at n={args.n_replicates}. Proceed with full sweep.")


if __name__ == "__main__":
    main()
