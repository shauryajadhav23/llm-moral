"""
Pydantic models for Scenario, Component, Path, and Manifest rows.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Optional

from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class IndependenceClass(str, Enum):
    commutative = "commutative"
    partial = "partial"
    entangled = "entangled"


class PathType(str, Enum):
    sequential = "sequential"
    skipped = "skipped"
    alt_grouping = "alt_grouping"
    direct = "direct"
    length_matched = "length_matched"


class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    complete = "complete"
    failed = "failed"
    refused = "refused"


# ---------------------------------------------------------------------------
# Scenario schema
# ---------------------------------------------------------------------------

class Consideration(BaseModel):
    text: str
    depends_on: list[str] = []


class Scenario(BaseModel):
    id: str
    type: str
    independence_class: IndependenceClass
    setup: str
    considerations: dict[str, Consideration]
    terminal_wordings: dict[str, str]

    @field_validator("independence_class", mode="before")
    @classmethod
    def _validate_independence_class(cls, v: str) -> str:
        valid = {e.value for e in IndependenceClass}
        if v not in valid:
            raise ValueError(f"independence_class must be one of {sorted(valid)}, got {v!r}")
        return v

    @field_validator("setup")
    @classmethod
    def _setup_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("setup must be non-empty")
        return v.strip()

    @field_validator("terminal_wordings")
    @classmethod
    def _at_least_one_wording(cls, v: dict) -> dict:
        if not v:
            raise ValueError("At least one terminal wording required")
        return v

    @model_validator(mode="after")
    def _validate_dag(self) -> "Scenario":
        ids = set(self.considerations.keys())

        # Referential integrity
        for comp_id, comp in self.considerations.items():
            for dep in comp.depends_on:
                if dep not in ids:
                    raise ValueError(
                        f"Component {comp_id!r} depends on {dep!r} which is not defined"
                    )

        # Cycle detection
        if _has_cycle(self.considerations):
            raise ValueError("Circular dependency detected in considerations DAG")

        # 3–5 components per scenario
        n = len(self.considerations)
        if not (3 <= n <= 5):
            raise ValueError(f"Scenario must have 3–5 considerations, found {n}")

        return self

    @property
    def component_ids(self) -> list[str]:
        return list(self.considerations.keys())

    @property
    def deps(self) -> dict[str, list[str]]:
        """Map from component ID to its list of direct predecessors."""
        return {k: v.depends_on for k, v in self.considerations.items()}


def _has_cycle(considerations: dict[str, Consideration]) -> bool:
    """Detect cycles in the dependency DAG using DFS colouring."""
    WHITE, GRAY, BLACK = 0, 1, 2
    colour = {k: WHITE for k in considerations}

    def dfs(node: str) -> bool:
        colour[node] = GRAY
        for dep in considerations[node].depends_on:
            if colour[dep] == GRAY:
                return True
            if colour[dep] == WHITE and dfs(dep):
                return True
        colour[node] = BLACK
        return False

    for node in considerations:
        if colour[node] == WHITE:
            if dfs(node):
                return True
    return False


# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------

class Path(BaseModel):
    path_signature: str
    turns: list[str]          # non-terminal user-turn strings (setup not included)
    perm_id: str              # 16-char deterministic hex hash
    n_turns: int              # len(turns), excludes the terminal wording
    components_used: list[str]  # component IDs in order of introduction
    path_type: PathType

    @classmethod
    def build(
        cls,
        turns: list[str],
        components_used: list[str],
        path_type: PathType,
        signature: str,
    ) -> "Path":
        raw = json.dumps(
            {"signature": signature, "turns": turns}, sort_keys=True
        )
        perm_id = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return cls(
            path_signature=signature,
            turns=turns,
            perm_id=perm_id,
            n_turns=len(turns),
            components_used=components_used,
            path_type=path_type,
        )


# ---------------------------------------------------------------------------
# Manifest row
# ---------------------------------------------------------------------------

class ManifestRow(BaseModel):
    run_id: str
    scenario_id: str
    path_type: str
    path_signature: str
    perm_id: str
    terminal_wording_id: str
    model: str
    provider: str
    temperature: float
    seed: int
    replicate_idx: int
    status: RunStatus = RunStatus.pending

    # Populated after execution
    verdict: Optional[int] = None       # 1 = yes, 0 = no, None = refused/pending
    model_version: Optional[str] = None
    provider_used: Optional[str] = None
    n_turns: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: Optional[int] = None
    timestamp: Optional[str] = None
