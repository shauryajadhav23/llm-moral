"""
Path generator: produces all valid conversation paths from a scenario.

Each PathType implements a distinct experimental condition:
  sequential    — cumulative one-at-a-time build-up respecting the DAG
  skipped       — single root component then terminal
  alt_grouping  — one grouped multi-component turn then terminal
  direct        — control: just the terminal (no preceding context)
  length_matched — neutral filler matched to sequential token count, then terminal
"""
from __future__ import annotations

import random
from itertools import combinations

from .schema import Path, PathType, Scenario

# ---------------------------------------------------------------------------
# Filler sentences (neutral, factually unrelated to moral content)
# ---------------------------------------------------------------------------

_FILLER_SENTENCES = [
    "The average depth of the Pacific Ocean is approximately 4,000 metres.",
    "Mount Everest has an elevation of 8,849 metres above sea level.",
    "The Amazon River discharges approximately 209,000 cubic metres of water per second into the Atlantic Ocean.",
    "Light travels at approximately 299,792 kilometres per second in a vacuum.",
    "The circumference of the Earth at the equator is approximately 40,075 kilometres.",
    "The Sahara Desert covers approximately 9.2 million square kilometres across northern Africa.",
    "The human body is estimated to contain approximately 37.2 trillion individual cells.",
    "The Milky Way galaxy contains an estimated 100 to 400 billion stars.",
    "The speed of sound in air at sea level is approximately 343 metres per second.",
    "The Great Barrier Reef extends approximately 2,300 kilometres along the Queensland coast.",
    "Lake Baikal in Siberia holds approximately 20 percent of the world's unfrozen surface fresh water.",
    "The core of the Earth maintains a temperature of approximately 5,100 degrees Celsius.",
    "The wingspan of an albatross can reach up to 3.7 metres, the largest of any living bird.",
    "Antarctica contains approximately 70 percent of the world's fresh water in the form of ice.",
    "The Dead Sea lies approximately 430 metres below sea level, the lowest point on Earth's surface.",
    "The average distance from the Earth to the Moon is approximately 384,400 kilometres.",
    "Photosynthesis converts approximately 130 terawatts of solar energy annually into chemical energy.",
    "The deepest point in the ocean, the Challenger Deep, reaches approximately 10,935 metres below sea level.",
    "A neutron star can rotate up to 716 times per second, the fastest known rotation of any stellar object.",
    "The Mariana Trench was first reached by humans in 1960 aboard the bathyscaphe Trieste.",
]

# ---------------------------------------------------------------------------
# DAG utilities
# ---------------------------------------------------------------------------

def _all_topological_sorts(
    component_ids: list[str],
    deps: dict[str, list[str]],
) -> list[list[str]]:
    """Return all valid topological orderings respecting `deps`.

    deps[node] = list of nodes that must appear before `node`.
    """
    results: list[list[str]] = []
    placed_set: set[str] = set()

    def can_place(node: str) -> bool:
        return all(d in placed_set for d in deps.get(node, []))

    def backtrack(placed: list[str], remaining: set[str]) -> None:
        if not remaining:
            results.append(placed[:])
            return
        available = sorted(n for n in remaining if can_place(n))
        for node in available:
            placed.append(node)
            placed_set.add(node)
            remaining.discard(node)
            backtrack(placed, remaining)
            placed.pop()
            placed_set.discard(node)
            remaining.add(node)

    backtrack([], set(component_ids))
    return results


def _closed_subsets(
    component_ids: list[str],
    deps: dict[str, list[str]],
    min_size: int = 2,
) -> list[frozenset[str]]:
    """Return all subsets S where every dependency of any member of S is also in S."""
    result: list[frozenset[str]] = []
    for size in range(min_size, len(component_ids)):
        for combo in combinations(component_ids, size):
            s = frozenset(combo)
            if all(d in s for c in combo for d in deps.get(c, [])):
                result.append(s)
    return result


# ---------------------------------------------------------------------------
# Token-count approximation
# ---------------------------------------------------------------------------

def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _make_filler(target_tokens: int) -> str:
    """Build neutral filler text reaching approximately target_tokens."""
    sentences: list[str] = []
    total = 0
    idx = 0
    while total < target_tokens:
        s = _FILLER_SENTENCES[idx % len(_FILLER_SENTENCES)]
        sentences.append(s)
        total += _approx_tokens(s)
        idx += 1
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Turn-text builders
# ---------------------------------------------------------------------------

def _component_turn(scenario: Scenario, comp_id: str, is_first: bool) -> str:
    text = scenario.considerations[comp_id].text.strip()
    if is_first:
        return f"Here is a consideration: {text}"
    return f"Here is an additional consideration: {text}"


def _group_turn(scenario: Scenario, comp_ids: list[str]) -> str:
    parts = [scenario.considerations[c].text.strip() for c in comp_ids]
    joined = " ".join(parts)
    return f"Here are several considerations: {joined}"


# ---------------------------------------------------------------------------
# Path type implementations
# ---------------------------------------------------------------------------

def _sequential_paths(scenario: Scenario) -> list[Path]:
    """One path per valid topological ordering; each turn adds one component."""
    orderings = _all_topological_sorts(scenario.component_ids, scenario.deps)
    paths: list[Path] = []

    for ordering in orderings:
        turns: list[str] = []
        cumulative: list[str] = []
        sig_parts: list[str] = []

        for i, comp_id in enumerate(ordering):
            turns.append(_component_turn(scenario, comp_id, is_first=(i == 0)))
            cumulative.append(comp_id)
            sig_parts.append("[" + ",".join(cumulative) + "]")

        signature = "".join(sig_parts) + "[F]"
        paths.append(Path.build(turns, ordering, PathType.sequential, signature))

    return paths


def _skipped_paths(scenario: Scenario) -> list[Path]:
    """[Ci][F] for each root component (no dependencies)."""
    paths: list[Path] = []
    for comp_id, comp in scenario.considerations.items():
        if comp.depends_on:
            continue  # only roots are valid starting points
        turn = _component_turn(scenario, comp_id, is_first=True)
        signature = f"[{comp_id}][F]"
        paths.append(Path.build([turn], [comp_id], PathType.skipped, signature))
    return paths


def _alt_grouping_paths(scenario: Scenario) -> list[Path]:
    """One grouped multi-component turn per topologically closed subset."""
    subsets = _closed_subsets(scenario.component_ids, scenario.deps, min_size=2)
    paths: list[Path] = []

    for subset in subsets:
        # Sort for determinism; topological order within the subset
        sub_orderings = _all_topological_sorts(list(subset), scenario.deps)
        if not sub_orderings:
            continue
        ordered = sub_orderings[0]  # pick first valid ordering for the label
        turn = _group_turn(scenario, ordered)
        sig_ids = ",".join(ordered)
        signature = f"[{sig_ids}][F]"
        paths.append(Path.build([turn], ordered, PathType.alt_grouping, signature))

    return paths


def _direct_paths(scenario: Scenario) -> list[Path]:
    """Control: no preceding context, just the terminal."""
    return [Path.build([], [], PathType.direct, "[F]")]


def _length_matched_paths(scenario: Scenario) -> list[Path]:
    """Filler turn(s) matching the token budget of the sequential paths."""
    seq_paths = _sequential_paths(scenario)
    if not seq_paths:
        return []

    # All sequential orderings of the same scenario have the same total token count
    # (same components, just reordered), so use the first one as the reference.
    ref_turns = seq_paths[0].turns
    target_tokens = sum(_approx_tokens(t) for t in ref_turns)

    filler = _make_filler(target_tokens)
    turn = f"For context: {filler}"
    signature = "[FILLER][F]"
    return [Path.build([turn], [], PathType.length_matched, signature)]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

_GENERATORS = {
    PathType.sequential: _sequential_paths,
    PathType.skipped: _skipped_paths,
    PathType.alt_grouping: _alt_grouping_paths,
    PathType.direct: _direct_paths,
    PathType.length_matched: _length_matched_paths,
}


def generate_paths(
    scenario: Scenario,
    path_type: PathType,
    seed: int = 0,
) -> list[Path]:
    """Return all valid paths for `scenario` under `path_type`.

    `seed` is accepted for interface compatibility but the generation is
    fully deterministic (no random sampling); ordering is sorted / canonical.
    """
    generator = _GENERATORS[path_type]
    paths = generator(scenario)
    # Sort by signature for stable ordering across calls
    return sorted(paths, key=lambda p: p.path_signature)
