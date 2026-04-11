"""
Secondary analyses (spec §10):
  1. Effect size by independence_class (commutative vs partial vs entangled)
  2. direct vs length_matched comparison (trajectory vs token-count confound)
  3. Refusal rate by path_type and wording
  4. Cross-wording stability of path-dependence effect

Usage:
    python analysis/secondary_tests.py --db outputs/runs.db [--plot]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

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
        r.total_tokens,
        s.independence_class
    FROM runs r
    JOIN scenarios s ON r.scenario_id = s.scenario_id
    WHERE r.status IN ('complete', 'refused')
    """
    return db.to_dataframe(sql)


# ---------------------------------------------------------------------------
# 1. Effect size by independence_class
# ---------------------------------------------------------------------------

def effect_by_independence_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare path_type effect magnitude across independence_class values.

    For each (independence_class, wording), compute the range of p_yes
    across path_types as a crude effect size (max - min).
    """
    complete = df[df["verdict"].notna()].copy()
    grp = complete.groupby(
        ["independence_class", "path_type", "terminal_wording_id"]
    )["verdict"].mean().reset_index()
    grp.columns = ["independence_class", "path_type", "terminal_wording_id", "p_yes"]

    result = []
    for (ic, wording), sub in grp.groupby(["independence_class", "terminal_wording_id"]):
        rng = sub["p_yes"].max() - sub["p_yes"].min()
        result.append({
            "independence_class": ic,
            "terminal_wording_id": wording,
            "p_yes_range": rng,
            "n_path_types": len(sub),
        })
    return pd.DataFrame(result).sort_values("independence_class")


# ---------------------------------------------------------------------------
# 2. direct vs length_matched (trajectory vs token-count confound)
# ---------------------------------------------------------------------------

def direct_vs_length_matched(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-proportion z-test comparing 'direct' and 'length_matched' per
    (scenario, wording) cell.

    A significant difference rules out token count as the sole driver.
    A non-significant difference suggests the path effect (if any) is not
    driven by token count alone.
    """
    complete = df[df["verdict"].notna()]
    subset = complete[complete["path_type"].isin(["direct", "length_matched"])]

    rows = []
    for (scenario_id, wording), sub in subset.groupby(["scenario_id", "terminal_wording_id"]):
        d = sub[sub["path_type"] == "direct"]["verdict"]
        lm = sub[sub["path_type"] == "length_matched"]["verdict"]
        if len(d) == 0 or len(lm) == 0:
            continue

        p_direct = d.mean()
        p_lm = lm.mean()
        # Two-proportion z-test
        n1, n2 = len(d), len(lm)
        p_pool = (d.sum() + lm.sum()) / (n1 + n2)
        if p_pool in (0, 1):
            z, p_val = float("nan"), float("nan")
        else:
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
            z = (p_direct - p_lm) / se if se > 0 else float("nan")
            p_val = 2 * stats.norm.sf(abs(z))

        rows.append({
            "scenario_id": scenario_id,
            "terminal_wording_id": wording,
            "n_direct": n1,
            "n_length_matched": n2,
            "p_yes_direct": p_direct,
            "p_yes_length_matched": p_lm,
            "z_stat": z,
            "p_value": p_val,
            "significant_005": p_val < 0.05 if not np.isnan(p_val) else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Refusal rate by path_type and wording
# ---------------------------------------------------------------------------

def refusal_rates(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["refused"] = (df2["status"] == "refused").astype(float)
    result = df2.groupby(["path_type", "terminal_wording_id"])["refused"].agg(
        n="count", refusal_rate="mean"
    ).reset_index()
    return result.sort_values(["path_type", "terminal_wording_id"])


# ---------------------------------------------------------------------------
# 4. Cross-wording stability
# ---------------------------------------------------------------------------

def cross_wording_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (scenario, path_type), compute pairwise correlation of p_yes
    across wordings. High correlation = path effect is wording-stable.
    """
    complete = df[df["verdict"].notna()]
    grp = complete.groupby(
        ["scenario_id", "path_type", "terminal_wording_id"]
    )["verdict"].mean().reset_index()
    grp.columns = ["scenario_id", "path_type", "terminal_wording_id", "p_yes"]

    rows = []
    wordings = grp["terminal_wording_id"].unique()
    for (scenario_id, path_type), sub in grp.groupby(["scenario_id", "path_type"]):
        pivot = sub.pivot(index="terminal_wording_id", columns="path_type", values="p_yes")
        # Pairwise correlation across wordings within this cell
        vals = sub.set_index("terminal_wording_id")["p_yes"]
        if len(vals) < 2:
            continue
        rows.append({
            "scenario_id": scenario_id,
            "path_type": path_type,
            "n_wordings": len(vals),
            "p_yes_mean": vals.mean(),
            "p_yes_std_across_wordings": vals.std(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Token count validation (verify length_matched control is matched)
# ---------------------------------------------------------------------------

def token_count_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare mean total_tokens for sequential vs length_matched per scenario."""
    subset = df[df["path_type"].isin(["sequential", "length_matched"])]
    result = subset.groupby(["scenario_id", "path_type"])["total_tokens"].agg(
        n="count", mean_tokens="mean", std_tokens="std"
    ).reset_index()
    return result.sort_values(["scenario_id", "path_type"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Secondary analyses")
    parser.add_argument("--db", default="outputs/runs.db")
    parser.add_argument("--plot", action="store_true", help="Generate plots (requires matplotlib)")
    args = parser.parse_args()

    with DB(args.db) as db:
        df = load_results(db)

    if df.empty:
        print("No runs found.")
        sys.exit(1)

    print(f"Loaded {len(df)} run records.\n")

    # 1. Effect by independence_class
    print("=== 1. Path-type effect range by independence_class ===")
    ic_df = effect_by_independence_class(df)
    print(ic_df.to_string(index=False))
    print(
        "\nExpected: larger p_yes_range for 'partial' and 'entangled' "
        "than 'commutative'."
    )

    # 2. direct vs length_matched
    print("\n=== 2. direct vs length_matched (trajectory vs token-count confound) ===")
    dlm_df = direct_vs_length_matched(df)
    if dlm_df.empty:
        print("Insufficient data for direct vs length_matched comparison.")
    else:
        print(dlm_df.to_string(index=False))
        sig = dlm_df["significant_005"].sum() if "significant_005" in dlm_df else 0
        print(
            f"\n{sig}/{len(dlm_df)} cells significant at p<0.05. "
            "Significant cells suggest trajectory beyond token-count."
        )

    # 3. Refusal rates
    print("\n=== 3. Refusal rate by path_type × wording ===")
    ref_df = refusal_rates(df)
    print(ref_df.to_string(index=False))

    # 4. Cross-wording stability
    print("\n=== 4. Cross-wording stability ===")
    stab_df = cross_wording_stability(df)
    if stab_df.empty:
        print("Insufficient data for cross-wording stability.")
    else:
        print(stab_df.to_string(index=False))
        print(
            "\nLow std_across_wordings → path effect is wording-stable. "
            "High std → sensitivity to wording framing."
        )

    # Token count check
    print("\n=== 5. Token count: sequential vs length_matched ===")
    tok_df = token_count_comparison(df)
    if tok_df.empty:
        print("No token count data available (runs not yet complete).")
    else:
        print(tok_df.to_string(index=False))

    # Optional plots
    if args.plot:
        _generate_plots(df)


def _generate_plots(df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed. Skipping plots.")
        return

    complete = df[df["verdict"].notna()]

    # Plot 1: Verdict proportion by path_type
    fig, ax = plt.subplots(figsize=(10, 5))
    grp = complete.groupby("path_type")["verdict"].mean().reset_index()
    sns.barplot(data=grp, x="path_type", y="verdict", ax=ax)
    ax.set_title("Mean verdict (p_yes) by path type")
    ax.set_ylabel("Proportion 'yes'")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    Path("outputs").mkdir(exist_ok=True)
    fig.savefig("outputs/verdict_by_path_type.png", dpi=150)
    print("Saved outputs/verdict_by_path_type.png")

    # Plot 2: Heatmap path_type × wording
    pivot = complete.groupby(["path_type", "terminal_wording_id"])["verdict"].mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
    ax.set_title("Mean verdict by path_type × wording")
    plt.tight_layout()
    fig.savefig("outputs/heatmap_path_wording.png", dpi=150)
    print("Saved outputs/heatmap_path_wording.png")

    plt.show()


if __name__ == "__main__":
    main()
