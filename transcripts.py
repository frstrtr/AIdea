"""Pipeline transcript log for AIdea.

Captures every event that passes through the pipeline:

  - request_started: the user's input — topic, mode, flags, source
    (cli / web / telegram-chat-id)
  - deck:             the donor deck that was loaded or generated
  - llm_call:         every LLM round-trip with full prompt + response
                      (deck-gen / synth / critic / refine / evolve)
  - llm_error:        each retry attempt's failure
  - sample:           per-idea, the K cards drawn from the deck
  - idea:             per-idea, the synthesizer's output text + mechanism
  - score:            per-idea, the critic's structured score
  - winner:           which idea won + total
  - refined:          the refined-winner text
  - evolved:          before / after card pairs after deck evolution
  - request_completed | request_errored

The log is structured JSONL (one JSON object per line) so it can be
loaded directly into pandas / DuckDB / jq for analysis. Records carry
``ts`` (unix seconds), ``run_id`` (from usage.current_run_id), ``source``
(set by the orchestration layer via ``set_source``), and a ``kind``
discriminator, plus per-kind payload fields.

Privacy: every prompt and response is captured verbatim. The log lives
on the same host that runs AIdea and is gitignored. The topic itself is
the user's intentional input.
"""

from __future__ import annotations

import contextvars
import json
import time
from pathlib import Path
from typing import Any

from usage import current_run_id


TRANSCRIPTS_LOG = Path(__file__).parent / "transcripts.jsonl"


# ---------------------------------------------------------------------------
# Source context: who is driving the request (cli / web / telegram)
# ---------------------------------------------------------------------------


_source_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "aidea_source", default="unknown",
)


def set_source(source: str) -> str:
    """Set the source label for the current run. Returns the value."""
    _source_var.set(source)
    return source


def current_source() -> str:
    return _source_var.get()


# ---------------------------------------------------------------------------
# Append
# ---------------------------------------------------------------------------


def _path() -> Path:
    TRANSCRIPTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    return TRANSCRIPTS_LOG


def log_event(kind: str, **payload: Any) -> None:
    """Append one structured event to transcripts.jsonl. Never raises."""
    record: dict[str, Any] = {
        "ts": time.time(),
        "run_id": current_run_id() or "ad-hoc",
        "source": current_source(),
        "kind": kind,
    }
    record.update(payload)
    try:
        with _path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        # Transcript logging is a best-effort sidecar; never block the pipeline.
        pass


# ---------------------------------------------------------------------------
# Read (for analysis utilities)
# ---------------------------------------------------------------------------


def read_events(
    *,
    run_id: str | None = None,
    source: str | None = None,
    kind: str | None = None,
) -> list[dict[str, Any]]:
    """Load events from the log, with optional filters."""
    if not TRANSCRIPTS_LOG.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with TRANSCRIPTS_LOG.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if run_id is not None and rec.get("run_id") != run_id:
                    continue
                if source is not None and rec.get("source") != source:
                    continue
                if kind is not None and rec.get("kind") != kind:
                    continue
                out.append(rec)
    except OSError:
        return []
    return out
