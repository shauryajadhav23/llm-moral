"""
Run orchestrator: manifest creation, shuffled async dispatch, and resume logic.

Workflow:
  1. Load scenarios
  2. Generate all (scenario, path_type, perm, wording, replicate) combinations
  3. Write the full manifest to the DB before any API calls
  4. Dispatch pending rows in shuffled order with a concurrency semaphore
  5. Update each row in-place after completion
"""
from __future__ import annotations

import asyncio
import logging
import random
import uuid
from typing import Any

from .adapter import Adapter
from .path_generator import generate_paths
from .schema import ManifestRow, PathType, RunStatus, Scenario
from .storage import DB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manifest generation
# ---------------------------------------------------------------------------

def build_manifest(
    scenarios: list[Scenario],
    model: str,
    provider: str,
    temperature: float,
    replicates: int,
    seed_base: int = 42,
) -> list[ManifestRow]:
    """Generate the complete experiment manifest (one row per run).

    No API calls are made here.
    """
    rows: list[ManifestRow] = []

    for scenario in scenarios:
        for path_type in PathType:
            paths = generate_paths(scenario, path_type, seed=seed_base)
            for path in paths:
                for wording_id in scenario.terminal_wordings:
                    for rep_idx in range(replicates):
                        # Deterministic seed per condition
                        condition_seed = abs(hash((
                            scenario.id,
                            path.perm_id,
                            wording_id,
                            rep_idx,
                        ))) % (2**31)
                        rows.append(ManifestRow(
                            run_id=str(uuid.uuid4()),
                            scenario_id=scenario.id,
                            path_type=path_type.value,
                            path_signature=path.path_signature,
                            perm_id=path.perm_id,
                            terminal_wording_id=wording_id,
                            model=model,
                            provider=provider,
                            temperature=temperature,
                            seed=condition_seed,
                            replicate_idx=rep_idx,
                            status=RunStatus.pending,
                        ))

    return rows


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class Orchestrator:
    def __init__(
        self,
        db: DB,
        adapter: Adapter,
        scenarios: list[Scenario],
        concurrency: int = 8,
    ) -> None:
        self.db = db
        self.adapter = adapter
        self.concurrency = concurrency
        # Build a lookup: scenario_id → Scenario
        self._scenarios: dict[str, Scenario] = {s.id: s for s in scenarios}

    async def run_all(self, manifest: list[ManifestRow]) -> None:
        """Dispatch all pending rows in shuffled order."""
        pending = [r for r in manifest if r.status == RunStatus.pending]
        random.shuffle(pending)

        logger.info(
            "Dispatching %d pending runs (concurrency=%d)", len(pending), self.concurrency
        )
        counts = self.db.count_by_status()
        logger.info("Manifest status: %s", counts)

        sem = asyncio.Semaphore(self.concurrency)

        async def _run_with_sem(row: ManifestRow) -> None:
            async with sem:
                await self._dispatch_one(row)

        await asyncio.gather(*[_run_with_sem(row) for row in pending])

        final_counts = self.db.count_by_status()
        logger.info("Completed. Status: %s", final_counts)

    async def _dispatch_one(self, row: ManifestRow) -> None:
        """Execute a single manifest row, update the DB."""
        scenario = self._scenarios.get(row.scenario_id)
        if scenario is None:
            logger.error("Unknown scenario_id %r for run %s", row.scenario_id, row.run_id)
            self.db.update_run(row.run_id, status=RunStatus.failed)
            return

        # Reconstruct the path from scenario + manifest metadata
        paths = generate_paths(
            scenario,
            PathType(row.path_type),
            seed=row.seed,
        )
        path = next((p for p in paths if p.perm_id == row.perm_id), None)
        if path is None:
            logger.error(
                "perm_id %r not found for scenario %s / %s",
                row.perm_id, row.scenario_id, row.path_type,
            )
            self.db.update_run(row.run_id, status=RunStatus.failed)
            return

        terminal_wording = scenario.terminal_wordings.get(row.terminal_wording_id)
        if terminal_wording is None:
            logger.error(
                "terminal_wording_id %r not found in scenario %s",
                row.terminal_wording_id, row.scenario_id,
            )
            self.db.update_run(row.run_id, status=RunStatus.failed)
            return

        # Mark as running
        self.db.update_run(row.run_id, status=RunStatus.running)

        try:
            result = await self.adapter.execute_conversation(
                scenario=scenario,
                path=path,
                terminal_wording=terminal_wording,
                seed=row.seed,
                run_id=row.run_id,
            )
        except Exception as exc:
            logger.exception("Run %s failed: %s", row.run_id, exc)
            self.db.update_run(row.run_id, status=RunStatus.failed)
            return

        # Persist turns
        self.db.insert_turns(result["turns"])

        # Update run row
        self.db.update_run(
            row.run_id,
            status=result["status"],
            verdict=result["verdict"],
            model_version=result["model_version"],
            provider_used=result["provider_used"],
            n_turns=path.n_turns,
            total_tokens=result["total_tokens"],
            latency_ms=result["latency_ms"],
        )

        logger.info(
            "run %s  scenario=%-30s  path=%-20s  wording=%s  rep=%02d  verdict=%s  status=%s",
            row.run_id[:8],
            row.scenario_id,
            row.path_signature,
            row.terminal_wording_id,
            row.replicate_idx,
            result["verdict"],
            result["status"].value,
        )
