"""
Scenario YAML loading, validation, and comprehension-check helpers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .schema import Scenario


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_scenario(path: Path) -> Scenario:
    """Load and validate a single scenario YAML file.

    Raises ValueError or pydantic.ValidationError on invalid input.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    if raw is None:
        raise ValueError(f"Empty YAML file: {path}")

    # Normalise considerations: inject the dict key as `id` is not stored in YAML
    raw_considerations = raw.get("considerations", {})
    if not isinstance(raw_considerations, dict):
        raise ValueError(f"'considerations' must be a mapping in {path}")

    try:
        scenario = Scenario(**raw)
    except ValidationError as exc:
        raise ValueError(f"Validation error in {path}:\n{exc}") from exc

    return scenario


def load_all_scenarios(scenarios_dir: Path) -> list[Scenario]:
    """Load every *.yaml file in scenarios_dir, enforcing unique IDs.

    Returns scenarios sorted by their id for determinism.
    Raises ValueError on any schema or uniqueness violation.
    """
    yaml_files = sorted(scenarios_dir.glob("*.yaml"))
    if not yaml_files:
        raise ValueError(f"No scenario YAML files found in {scenarios_dir}")

    scenarios: list[Scenario] = []
    seen_ids: set[str] = set()
    errors: list[str] = []

    for yaml_file in yaml_files:
        try:
            scenario = load_scenario(yaml_file)
        except (ValueError, Exception) as exc:
            errors.append(f"  {yaml_file.name}: {exc}")
            continue

        if scenario.id in seen_ids:
            errors.append(f"  {yaml_file.name}: duplicate scenario ID {scenario.id!r}")
            continue

        seen_ids.add(scenario.id)
        scenarios.append(scenario)

    if errors:
        raise ValueError("Scenario loading failed:\n" + "\n".join(errors))

    return sorted(scenarios, key=lambda s: s.id)


# ---------------------------------------------------------------------------
# Corpus-level checks
# ---------------------------------------------------------------------------

_MIN_SCENARIOS = 5
_REQUIRED_TYPES = {"utilitarian_tradeoff", "competing_obligations"}


def validate_corpus(scenarios: list[Scenario]) -> None:
    """Enforce corpus-level requirements from the spec.

    Raises ValueError listing all violations found.
    """
    violations: list[str] = []

    if len(scenarios) < _MIN_SCENARIOS:
        violations.append(
            f"Corpus must contain at least {_MIN_SCENARIOS} scenarios; "
            f"found {len(scenarios)}"
        )

    types_present = {s.type for s in scenarios}
    missing_types = _REQUIRED_TYPES - types_present
    if missing_types:
        violations.append(
            f"Corpus missing required scenario types: {sorted(missing_types)}"
        )

    independence_classes = {s.independence_class for s in scenarios}
    if len(independence_classes) < 2:
        violations.append(
            "Corpus must use at least 2 different independence_class values; "
            f"found only: {independence_classes}"
        )

    if violations:
        raise ValueError("Corpus validation failed:\n" + "\n".join(f"  - {v}" for v in violations))


# ---------------------------------------------------------------------------
# Comprehension check helper (for human review)
# ---------------------------------------------------------------------------

def print_comprehension_report(scenarios: list[Scenario]) -> None:
    """Pretty-print a summary of each scenario's paths for manual review."""
    from .path_generator import generate_paths
    from .schema import PathType

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario.id}  [{scenario.type} / {scenario.independence_class}]")
        print(f"Setup: {scenario.setup[:120].strip()}...")
        for wording_id, wording in scenario.terminal_wordings.items():
            print(f"  {wording_id}: {wording}")

        for pt in PathType:
            paths = generate_paths(scenario, pt, seed=0)
            print(f"\n  [{pt.value}] — {len(paths)} path(s)")
            for p in paths[:3]:  # show at most 3 per type
                print(f"    sig: {p.path_signature}")
                for i, turn in enumerate(p.turns):
                    print(f"    turn {i+1}: {turn[:80].strip()}")
