"""
Pipeline entrypoint.

Commands:
  generate   — build the full manifest and write it to DB (no API calls)
  run        — dispatch all pending manifest rows
  pilot      — run a subset (one scenario, configurable replicates) for piloting
  validate   — validate scenario YAML files and print a comprehension report
  status     — print current manifest status from DB
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: make `src` importable when invoked as python src/main.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapter import Adapter
from src.orchestrator import Orchestrator, build_manifest
from src.schema import PathType
from src.storage import DB
from src.validation import load_all_scenarios, validate_corpus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="moral-pipeline",
        description="LLM moral judgment path-dependence benchmark pipeline",
    )
    p.add_argument(
        "--scenarios-dir",
        default="scenarios",
        help="Directory containing scenario YAML files (default: scenarios/)",
    )
    p.add_argument(
        "--db-path",
        default=os.getenv("DB_PATH", "outputs/runs.db"),
        help="DuckDB database file path",
    )
    p.add_argument(
        "--model",
        default=os.getenv("MODEL", "openrouter/openai/gpt-oss-120b"),
        help="LiteLLM model string",
    )
    p.add_argument(
        "--provider",
        default=os.getenv("PROVIDER", "Fireworks"),
        help="OpenRouter inference provider to pin",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("TEMPERATURE", "0.7")),
        help="Sampling temperature",
    )
    p.add_argument(
        "--replicates",
        type=int,
        default=int(os.getenv("REPLICATES", "30")),
        help="Replicates per condition",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("CONCURRENCY", "8")),
        help="Max parallel API requests",
    )
    p.add_argument(
        "--cache-dir",
        default=os.getenv("CACHE_DIR", "outputs/cache"),
        help="Directory for file-based response cache",
    )

    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("generate", help="Generate manifest only (no API calls)")

    run_p = sub.add_parser("run", help="Dispatch all pending manifest rows")
    run_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be dispatched without making API calls",
    )

    pilot_p = sub.add_parser("pilot", help="Pilot run on a single scenario")
    pilot_p.add_argument(
        "--scenario-id",
        required=True,
        help="Scenario ID to pilot",
    )
    pilot_p.add_argument(
        "--pilot-replicates",
        type=int,
        default=5,
        help="Replicates per condition for pilot (default: 5)",
    )
    pilot_p.add_argument(
        "--path-type",
        default="sequential",
        choices=[pt.value for pt in PathType],
        help="Path type to pilot",
    )

    sub.add_parser("validate", help="Validate scenario YAML files and print comprehension report")
    sub.add_parser("status", help="Print current manifest status from DB")

    return p


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> None:
    scenarios_dir = Path(args.scenarios_dir)
    logger.info("Loading scenarios from %s", scenarios_dir)
    scenarios = load_all_scenarios(scenarios_dir)
    validate_corpus(scenarios)
    logger.info("All %d scenarios valid.", len(scenarios))
    print(f"\nLoaded {len(scenarios)} scenarios:")
    for s in scenarios:
        print(
            f"  {s.id:<40}  type={s.type:<25}  "
            f"class={s.independence_class.value:<12}  "
            f"components={len(s.considerations)}"
        )

    from src.validation import print_comprehension_report
    print_comprehension_report(scenarios)


def cmd_generate(args: argparse.Namespace) -> None:
    scenarios_dir = Path(args.scenarios_dir)
    scenarios = load_all_scenarios(scenarios_dir)
    validate_corpus(scenarios)

    with DB(args.db_path) as db:
        for scenario in scenarios:
            db.upsert_scenario(scenario)

        manifest = build_manifest(
            scenarios=scenarios,
            model=args.model,
            provider=args.provider,
            temperature=args.temperature,
            replicates=args.replicates,
        )
        db.insert_manifest(manifest)

        counts = db.count_by_status()
        total = sum(counts.values())
        logger.info(
            "Manifest written: %d total rows | %s", total, counts
        )
        print(f"\nManifest summary: {total} runs planned.")
        for status, n in sorted(counts.items()):
            print(f"  {status}: {n}")


def cmd_run(args: argparse.Namespace) -> None:
    scenarios_dir = Path(args.scenarios_dir)
    scenarios = load_all_scenarios(scenarios_dir)

    with DB(args.db_path) as db:
        pending = db.get_pending_runs()
        if not pending:
            logger.info("No pending runs. Use 'generate' first, or all runs already complete.")
            return

        if args.dry_run:
            print(f"Dry run: would dispatch {len(pending)} pending rows.")
            return

        adapter = Adapter(
            model=args.model,
            provider=args.provider,
            temperature=args.temperature,
            cache_dir=args.cache_dir,
        )
        orchestrator = Orchestrator(
            db=db,
            adapter=adapter,
            scenarios=scenarios,
            concurrency=args.concurrency,
        )

        # Reconstruct ManifestRow objects from DB dicts for type-checking
        from src.schema import ManifestRow, RunStatus
        manifest = [
            ManifestRow(
                run_id=r["run_id"],
                scenario_id=r["scenario_id"],
                path_type=r["path_type"],
                path_signature=r["path_signature"],
                perm_id=r["perm_id"],
                terminal_wording_id=r["terminal_wording_id"],
                model=r["model"],
                provider=r["provider_requested"],
                temperature=r["temperature"],
                seed=r["seed"],
                replicate_idx=r["replicate_idx"],
                status=RunStatus(r["status"]),
            )
            for r in pending
        ]

        asyncio.run(orchestrator.run_all(manifest))


def cmd_pilot(args: argparse.Namespace) -> None:
    scenarios_dir = Path(args.scenarios_dir)
    scenarios = load_all_scenarios(scenarios_dir)

    target = next((s for s in scenarios if s.id == args.scenario_id), None)
    if target is None:
        logger.error("Scenario %r not found.", args.scenario_id)
        sys.exit(1)

    with DB(args.db_path) as db:
        db.upsert_scenario(target)

        pilot_manifest = build_manifest(
            scenarios=[target],
            model=args.model,
            provider=args.provider,
            temperature=args.temperature,
            replicates=args.pilot_replicates,
        )
        # Filter to one path type if specified
        if args.path_type:
            pilot_manifest = [
                r for r in pilot_manifest if r.path_type == args.path_type
            ]

        db.insert_manifest(pilot_manifest)
        logger.info("Pilot manifest: %d runs", len(pilot_manifest))

        adapter = Adapter(
            model=args.model,
            provider=args.provider,
            temperature=args.temperature,
            cache_dir=args.cache_dir,
        )
        orchestrator = Orchestrator(
            db=db,
            adapter=adapter,
            scenarios=[target],
            concurrency=args.concurrency,
        )
        asyncio.run(orchestrator.run_all(pilot_manifest))


def cmd_status(args: argparse.Namespace) -> None:
    with DB(args.db_path) as db:
        counts = db.count_by_status()
        if not counts:
            print("No manifest found. Run 'generate' first.")
            return
        total = sum(counts.values())
        print(f"\nManifest status ({total} total runs):")
        for status, n in sorted(counts.items()):
            pct = 100 * n / total if total else 0
            print(f"  {status:<12} {n:>6}  ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "validate": cmd_validate,
        "generate": cmd_generate,
        "run": cmd_run,
        "pilot": cmd_pilot,
        "status": cmd_status,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
