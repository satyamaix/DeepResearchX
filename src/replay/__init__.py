"""Deterministic Replay System for DRX Deep Research.

This module provides the replay infrastructure for debugging, training data
generation, and reproducible research runs. It enables recording and replaying
research sessions with full determinism.

Key Components:
- EventRecorder: Records events to the database for later replay
- ReplayPlayer: Replays sessions from checkpoints with optional modifications
- TypedDict definitions: LangGraph-compatible data structures

Usage:
    from src.replay import EventRecorder, ReplayPlayer, ReplayEvent

    # Recording events
    recorder = EventRecorder()
    await recorder.record_event(session_id, event)

    # Replaying sessions
    player = ReplayPlayer()
    async for event in player.replay_from_checkpoint(session_id, checkpoint_id):
        process(event)
"""

from __future__ import annotations

from src.replay.recorder import (
    EventRecorder,
    LLMCallRecord,
    ReplayEvent,
    ToolCallRecord,
)
from src.replay.player import (
    DiffReport,
    FieldDiff,
    ReplayPlayer,
)

__all__ = [
    # Recorder
    "EventRecorder",
    "ReplayEvent",
    "ToolCallRecord",
    "LLMCallRecord",
    # Player
    "ReplayPlayer",
    "DiffReport",
    "FieldDiff",
]
