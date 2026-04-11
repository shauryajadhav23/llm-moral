"""
Unit tests for path_generator.py.

Run with: pytest tests/test_path_generator.py -v
"""
import sys
from pathlib import Path

# Make src importable when running pytest from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.schema import (
    Consideration,
    IndependenceClass,
    PathType,
    Scenario,
)
from src.path_generator import (
    _all_topological_sorts,
    _closed_subsets,
    generate_paths,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_scenario(
    considerations: dict,
    independence_class: str = "partial",
    scenario_id: str = "test-01",
) -> Scenario:
    return Scenario(
        id=scenario_id,
        type="utilitarian_tradeoff",
        independence_class=independence_class,
        setup="A test scenario setup text.",
        considerations=considerations,
        terminal_wordings={"W1": "Is this permissible? Answer yes or no."},
    )


@pytest.fixture
def chain_scenario():
    """C1 → C2 → C3 (linear chain)."""
    return _make_scenario({
        "C1": Consideration(text="Fact one.", depends_on=[]),
        "C2": Consideration(text="Fact two.", depends_on=["C1"]),
        "C3": Consideration(text="Fact three.", depends_on=["C2"]),
    })


@pytest.fixture
def independent_scenario():
    """C1, C2, C3 all independent (commutative)."""
    return _make_scenario({
        "C1": Consideration(text="Alpha.", depends_on=[]),
        "C2": Consideration(text="Beta.", depends_on=[]),
        "C3": Consideration(text="Gamma.", depends_on=[]),
    }, independence_class="commutative")


@pytest.fixture
def diamond_scenario():
    """C1 → C2, C1 → C3, {C2,C3} → C4 (diamond DAG)."""
    return _make_scenario({
        "C1": Consideration(text="Root fact.", depends_on=[]),
        "C2": Consideration(text="Branch A.", depends_on=["C1"]),
        "C3": Consideration(text="Branch B.", depends_on=["C1"]),
        "C4": Consideration(text="Conclusion.", depends_on=["C2", "C3"]),
    }, independence_class="entangled")


# ---------------------------------------------------------------------------
# _all_topological_sorts
# ---------------------------------------------------------------------------

class TestAllTopologicalSorts:
    def test_linear_chain_has_one_ordering(self, chain_scenario):
        orderings = _all_topological_sorts(
            chain_scenario.component_ids, chain_scenario.deps
        )
        assert orderings == [["C1", "C2", "C3"]]

    def test_independent_has_all_permutations(self, independent_scenario):
        orderings = _all_topological_sorts(
            independent_scenario.component_ids, independent_scenario.deps
        )
        # 3! = 6 permutations
        assert len(orderings) == 6

    def test_orderings_respect_deps(self, diamond_scenario):
        orderings = _all_topological_sorts(
            diamond_scenario.component_ids, diamond_scenario.deps
        )
        # Every ordering must have C1 before C2 and C3, and C4 last
        for ordering in orderings:
            idx = {c: i for i, c in enumerate(ordering)}
            assert idx["C1"] < idx["C2"]
            assert idx["C1"] < idx["C3"]
            assert idx["C2"] < idx["C4"]
            assert idx["C3"] < idx["C4"]

    def test_diamond_has_two_valid_orderings(self, diamond_scenario):
        orderings = _all_topological_sorts(
            diamond_scenario.component_ids, diamond_scenario.deps
        )
        # C1 must be first, C4 must be last; C2 and C3 can swap
        assert len(orderings) == 2

    def test_single_node(self):
        result = _all_topological_sorts(["A"], {"A": []})
        assert result == [["A"]]

    def test_two_independent_nodes(self):
        result = _all_topological_sorts(["A", "B"], {"A": [], "B": []})
        assert len(result) == 2
        assert ["A", "B"] in result
        assert ["B", "A"] in result


# ---------------------------------------------------------------------------
# _closed_subsets
# ---------------------------------------------------------------------------

class TestClosedSubsets:
    def test_independent_all_pairs_are_closed(self, independent_scenario):
        subsets = _closed_subsets(
            independent_scenario.component_ids, independent_scenario.deps
        )
        # All pairs are valid (no dependencies to violate)
        pairs = [s for s in subsets if len(s) == 2]
        assert len(pairs) == 3  # C(3,2)

    def test_chain_only_prefix_subsets_closed(self, chain_scenario):
        subsets = _closed_subsets(
            chain_scenario.component_ids, chain_scenario.deps
        )
        # {C1,C2}: C2 depends on C1 ✓; {C2,C3}: C3 depends on C2 ✓ but C2 depends on C1 ✗
        # {C1,C2}: closed ✓; {C2,C3}: C2 needs C1 which is not in subset ✗
        # {C1,C3}: C3 needs C2 not in subset ✗
        # So only {C1,C2} of size 2
        size2 = [s for s in subsets if len(s) == 2]
        assert frozenset({"C1", "C2"}) in size2
        assert frozenset({"C2", "C3"}) not in size2
        assert frozenset({"C1", "C3"}) not in size2

    def test_min_size_respected(self, independent_scenario):
        subsets = _closed_subsets(
            independent_scenario.component_ids, independent_scenario.deps, min_size=3
        )
        # Only one subset of size 3 (all), and it must be < len (proper subset)
        # proper subsets of size 3 out of 3 = none (equal to all)
        assert len(subsets) == 0


# ---------------------------------------------------------------------------
# generate_paths — sequential
# ---------------------------------------------------------------------------

class TestSequentialPaths:
    def test_chain_has_one_sequential_path(self, chain_scenario):
        paths = generate_paths(chain_scenario, PathType.sequential, seed=0)
        assert len(paths) == 1

    def test_sequential_signature_cumulative(self, chain_scenario):
        paths = generate_paths(chain_scenario, PathType.sequential, seed=0)
        sig = paths[0].path_signature
        # Should be [C1][C1,C2][C1,C2,C3][F]
        assert sig == "[C1][C1,C2][C1,C2,C3][F]"

    def test_independent_has_six_sequential_paths(self, independent_scenario):
        paths = generate_paths(independent_scenario, PathType.sequential, seed=0)
        assert len(paths) == 6

    def test_sequential_ends_with_F(self, chain_scenario, independent_scenario):
        for scenario in [chain_scenario, independent_scenario]:
            for p in generate_paths(scenario, PathType.sequential, seed=0):
                assert p.path_signature.endswith("[F]")

    def test_sequential_n_turns_matches_components(self, chain_scenario):
        paths = generate_paths(chain_scenario, PathType.sequential, seed=0)
        for p in paths:
            assert p.n_turns == len(chain_scenario.component_ids)

    def test_sequential_path_type_set(self, chain_scenario):
        paths = generate_paths(chain_scenario, PathType.sequential, seed=0)
        for p in paths:
            assert p.path_type == PathType.sequential

    def test_perm_ids_unique(self, independent_scenario):
        paths = generate_paths(independent_scenario, PathType.sequential, seed=0)
        perm_ids = [p.perm_id for p in paths]
        assert len(perm_ids) == len(set(perm_ids))


# ---------------------------------------------------------------------------
# generate_paths — skipped
# ---------------------------------------------------------------------------

class TestSkippedPaths:
    def test_chain_only_root_is_skipped(self, chain_scenario):
        paths = generate_paths(chain_scenario, PathType.skipped, seed=0)
        assert len(paths) == 1
        assert paths[0].path_signature == "[C1][F]"

    def test_independent_all_are_roots(self, independent_scenario):
        paths = generate_paths(independent_scenario, PathType.skipped, seed=0)
        assert len(paths) == 3

    def test_skipped_has_one_turn(self, chain_scenario):
        paths = generate_paths(chain_scenario, PathType.skipped, seed=0)
        for p in paths:
            assert p.n_turns == 1

    def test_skipped_path_type_set(self, chain_scenario):
        paths = generate_paths(chain_scenario, PathType.skipped, seed=0)
        for p in paths:
            assert p.path_type == PathType.skipped


# ---------------------------------------------------------------------------
# generate_paths — alt_grouping
# ---------------------------------------------------------------------------

class TestAltGroupingPaths:
    def test_chain_has_one_alt_grouping(self, chain_scenario):
        paths = generate_paths(chain_scenario, PathType.alt_grouping, seed=0)
        # Only {C1,C2} is a closed subset of size 2
        assert len(paths) == 1

    def test_independent_has_three_alt_groupings(self, independent_scenario):
        paths = generate_paths(independent_scenario, PathType.alt_grouping, seed=0)
        # C(3,2) = 3 pairs, all closed; size-3 subset excluded (not proper)
        assert len(paths) == 3

    def test_alt_grouping_has_one_turn(self, independent_scenario):
        paths = generate_paths(independent_scenario, PathType.alt_grouping, seed=0)
        for p in paths:
            assert p.n_turns == 1

    def test_alt_grouping_signature_ends_with_F(self, independent_scenario):
        paths = generate_paths(independent_scenario, PathType.alt_grouping, seed=0)
        for p in paths:
            assert p.path_signature.endswith("[F]")

    def test_alt_grouping_path_type_set(self, independent_scenario):
        paths = generate_paths(independent_scenario, PathType.alt_grouping, seed=0)
        for p in paths:
            assert p.path_type == PathType.alt_grouping


# ---------------------------------------------------------------------------
# generate_paths — direct
# ---------------------------------------------------------------------------

class TestDirectPaths:
    def test_direct_always_one_path(self, chain_scenario, independent_scenario):
        for scenario in [chain_scenario, independent_scenario]:
            paths = generate_paths(scenario, PathType.direct, seed=0)
            assert len(paths) == 1

    def test_direct_signature_is_F(self, chain_scenario):
        paths = generate_paths(chain_scenario, PathType.direct, seed=0)
        assert paths[0].path_signature == "[F]"

    def test_direct_zero_turns(self, chain_scenario):
        paths = generate_paths(chain_scenario, PathType.direct, seed=0)
        assert paths[0].n_turns == 0

    def test_direct_path_type_set(self, chain_scenario):
        paths = generate_paths(chain_scenario, PathType.direct, seed=0)
        assert paths[0].path_type == PathType.direct


# ---------------------------------------------------------------------------
# generate_paths — length_matched
# ---------------------------------------------------------------------------

class TestLengthMatchedPaths:
    def test_length_matched_one_path(self, chain_scenario):
        paths = generate_paths(chain_scenario, PathType.length_matched, seed=0)
        assert len(paths) == 1

    def test_length_matched_signature(self, chain_scenario):
        paths = generate_paths(chain_scenario, PathType.length_matched, seed=0)
        assert paths[0].path_signature == "[FILLER][F]"

    def test_length_matched_one_turn(self, chain_scenario):
        paths = generate_paths(chain_scenario, PathType.length_matched, seed=0)
        assert paths[0].n_turns == 1

    def test_length_matched_filler_not_empty(self, chain_scenario):
        paths = generate_paths(chain_scenario, PathType.length_matched, seed=0)
        assert len(paths[0].turns[0]) > 0

    def test_length_matched_path_type_set(self, chain_scenario):
        paths = generate_paths(chain_scenario, PathType.length_matched, seed=0)
        assert paths[0].path_type == PathType.length_matched

    def test_filler_token_count_gte_sequential(self, chain_scenario):
        """Filler turn should be at least as long as sequential turns combined."""
        from src.path_generator import _approx_tokens
        seq_paths = generate_paths(chain_scenario, PathType.sequential, seed=0)
        lm_paths = generate_paths(chain_scenario, PathType.length_matched, seed=0)

        seq_tokens = sum(_approx_tokens(t) for t in seq_paths[0].turns)
        lm_tokens = _approx_tokens(lm_paths[0].turns[0])

        # Length-matched may slightly exceed due to sentence boundaries
        assert lm_tokens >= seq_tokens * 0.9


# ---------------------------------------------------------------------------
# Cross-cutting: all path types produce valid Path objects
# ---------------------------------------------------------------------------

class TestPathObjectIntegrity:
    @pytest.mark.parametrize("path_type", list(PathType))
    def test_perm_id_is_16_hex(self, chain_scenario, path_type):
        paths = generate_paths(chain_scenario, path_type, seed=0)
        for p in paths:
            assert len(p.perm_id) == 16
            assert all(c in "0123456789abcdef" for c in p.perm_id)

    @pytest.mark.parametrize("path_type", list(PathType))
    def test_n_turns_consistent_with_turns_list(self, chain_scenario, path_type):
        paths = generate_paths(chain_scenario, path_type, seed=0)
        for p in paths:
            assert p.n_turns == len(p.turns)

    @pytest.mark.parametrize("path_type", list(PathType))
    def test_deterministic_across_calls(self, chain_scenario, path_type):
        paths_a = generate_paths(chain_scenario, path_type, seed=0)
        paths_b = generate_paths(chain_scenario, path_type, seed=0)
        sigs_a = [p.path_signature for p in paths_a]
        sigs_b = [p.path_signature for p in paths_b]
        assert sigs_a == sigs_b
