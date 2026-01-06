"""
DRX Models Package - Data models for the Deep Research System.

This package provides TypedDict definitions and classes for:
- Knowledge Graph: Entities, Relations, and Claims
- Graph operations and export formats
"""

from .knowledge import (
    # Type aliases
    EntityType,
    ClaimStatus,
    # TypedDicts
    Entity,
    Relation,
    Claim,
    # Classes
    KnowledgeGraph,
    # Factory functions
    create_entity,
    create_relation,
    create_claim,
)

__all__ = [
    # Type aliases
    "EntityType",
    "ClaimStatus",
    # TypedDicts
    "Entity",
    "Relation",
    "Claim",
    # Classes
    "KnowledgeGraph",
    # Factory functions
    "create_entity",
    "create_relation",
    "create_claim",
]
