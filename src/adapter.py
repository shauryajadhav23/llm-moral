"""
Model adapter: LiteLLM wrapper targeting OpenRouter with pinned provider.

Responsibilities:
  - Execute a full multi-turn conversation from a Path
  - Extract a binary yes/no verdict from the terminal response
  - File-based response cache (keyed on full payload hash)
  - Logging of token counts, latency, and provider metadata
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import litellm

from .schema import ManifestRow, Path as ConvPath, RunStatus, Scenario

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System instruction appended to the setup text
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = (
    "\n\nYou will be presented with a series of moral considerations one at a time. "
    "Acknowledge each consideration briefly and neutrally. "
    "When asked a direct yes/no question, answer with only 'yes' or 'no' followed "
    "by a single sentence of reasoning. Do not hedge or refuse to answer."
)

# ---------------------------------------------------------------------------
# Verdict extraction
# ---------------------------------------------------------------------------

def extract_verdict(response_text: str) -> Optional[int]:
    """Parse binary verdict from model output.

    Returns 1 for yes, 0 for no, None if not parseable.
    """
    text = response_text.strip().lower()
    # First word heuristic
    first_word = text.split()[0].rstrip(".,;:") if text else ""
    if first_word == "yes":
        return 1
    if first_word == "no":
        return 0
    # Fallback: search for explicit yes/no anywhere near the start
    if text.startswith("yes"):
        return 1
    if text.startswith("no"):
        return 0
    logger.warning("Unparseable verdict response: %r", response_text[:120])
    return None


# ---------------------------------------------------------------------------
# File-based cache
# ---------------------------------------------------------------------------

def _cache_key(
    model: str,
    provider: str,
    messages: list[dict],
    temperature: float,
    seed: int,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "provider": provider,
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def _cache_get(cache_dir: Path, key: str) -> Optional[dict]:
    p = _cache_path(cache_dir, key)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _cache_set(cache_dir: Path, key: str, value: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_path(cache_dir, key).write_text(json.dumps(value))


# ---------------------------------------------------------------------------
# Core conversation executor
# ---------------------------------------------------------------------------

class Adapter:
    def __init__(
        self,
        model: str = "openrouter/openai/gpt-oss-120b",
        provider: str = "Fireworks",
        temperature: float = 0.7,
        cache_dir: str | Path = "outputs/cache",
        num_retries: int = 3,
    ) -> None:
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.cache_dir = Path(cache_dir)
        self.num_retries = num_retries
        self._api_key = os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            raise EnvironmentError("OPENROUTER_API_KEY environment variable not set")

    def _call(
        self,
        messages: list[dict],
        seed: int,
    ) -> dict[str, Any]:
        """Make one LiteLLM call with caching, retry, and metadata capture."""
        key = _cache_key(self.model, self.provider, messages, self.temperature, seed)
        cached = _cache_get(self.cache_dir, key)
        if cached is not None:
            logger.debug("Cache hit for key %s", key[:12])
            return cached

        t0 = time.monotonic()
        response = litellm.completion(
            model=self.model,
            messages=messages,
            api_key=self._api_key,
            temperature=self.temperature,
            seed=seed,
            num_retries=self.num_retries,
            extra_body={
                "provider": {
                    "order": [self.provider],
                    "allow_fallbacks": False,
                }
            },
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        content = response.choices[0].message.content or ""
        provider_used = (
            getattr(response, "model", None)
            or getattr(getattr(response, "_hidden_params", None), "response_ms", None)
        )
        # OpenRouter returns provider info in model field or headers
        try:
            provider_used = response._hidden_params.get("additional_headers", {}).get(
                "x-openrouter-provider", self.provider
            )
        except Exception:
            provider_used = self.provider

        usage = getattr(response, "usage", None)
        total_tokens = (usage.total_tokens if usage else 0) or 0
        model_version = getattr(response, "model", self.model)

        result = {
            "content": content,
            "latency_ms": latency_ms,
            "total_tokens": total_tokens,
            "provider_used": provider_used,
            "model_version": model_version,
            "raw_response": response.model_dump_json() if hasattr(response, "model_dump_json") else str(response),
        }
        _cache_set(self.cache_dir, key, result)
        return result

    async def execute_conversation(
        self,
        scenario: Scenario,
        path: ConvPath,
        terminal_wording: str,
        seed: int,
        run_id: str,
    ) -> dict[str, Any]:
        """Execute a full multi-turn conversation and return result metadata.

        Returns a dict with keys: verdict, status, turns, total_tokens,
        latency_ms, provider_used, model_version.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._execute_sync,
            scenario,
            path,
            terminal_wording,
            seed,
            run_id,
        )

    def _execute_sync(
        self,
        scenario: Scenario,
        path: ConvPath,
        terminal_wording: str,
        seed: int,
        run_id: str,
    ) -> dict[str, Any]:
        messages: list[dict] = [
            {"role": "system", "content": scenario.setup.strip() + SYSTEM_INSTRUCTION}
        ]
        turn_records: list[dict] = [
            {
                "turn_id": f"{run_id}_t0",
                "run_id": run_id,
                "turn_index": 0,
                "role": "system",
                "content": messages[0]["content"],
                "is_terminal": False,
                "raw_response": None,
            }
        ]

        cumulative_tokens = 0
        cumulative_latency = 0
        provider_used = self.provider
        model_version = self.model

        # Non-terminal turns
        for i, user_turn in enumerate(path.turns):
            messages.append({"role": "user", "content": user_turn})
            turn_records.append({
                "turn_id": f"{run_id}_t{len(turn_records)}",
                "run_id": run_id,
                "turn_index": len(turn_records),
                "role": "user",
                "content": user_turn,
                "is_terminal": False,
                "raw_response": None,
            })

            result = self._call(messages, seed=seed + i)
            cumulative_tokens += result["total_tokens"]
            cumulative_latency += result["latency_ms"]
            provider_used = result["provider_used"]
            model_version = result["model_version"]

            assistant_content = result["content"]
            messages.append({"role": "assistant", "content": assistant_content})
            turn_records.append({
                "turn_id": f"{run_id}_t{len(turn_records)}",
                "run_id": run_id,
                "turn_index": len(turn_records),
                "role": "assistant",
                "content": assistant_content,
                "is_terminal": False,
                "raw_response": result["raw_response"],
            })

        # Terminal turn
        messages.append({"role": "user", "content": terminal_wording})
        turn_records.append({
            "turn_id": f"{run_id}_t{len(turn_records)}",
            "run_id": run_id,
            "turn_index": len(turn_records),
            "role": "user",
            "content": terminal_wording,
            "is_terminal": True,
            "raw_response": None,
        })

        terminal_result = self._call(messages, seed=seed + len(path.turns))
        cumulative_tokens += terminal_result["total_tokens"]
        cumulative_latency += terminal_result["latency_ms"]
        provider_used = terminal_result["provider_used"]
        model_version = terminal_result["model_version"]

        terminal_content = terminal_result["content"]
        turn_records.append({
            "turn_id": f"{run_id}_t{len(turn_records)}",
            "run_id": run_id,
            "turn_index": len(turn_records),
            "role": "assistant",
            "content": terminal_content,
            "is_terminal": True,
            "raw_response": terminal_result["raw_response"],
        })

        verdict = extract_verdict(terminal_content)
        status = RunStatus.refused if verdict is None else RunStatus.complete

        return {
            "verdict": verdict,
            "status": status,
            "turns": turn_records,
            "total_tokens": cumulative_tokens,
            "latency_ms": cumulative_latency,
            "provider_used": provider_used,
            "model_version": model_version,
        }
