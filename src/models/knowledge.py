"""
Knowledge Graph Models for DRX Deep Research System.

Provides TypedDict definitions for entities, relations, and claims
that form the knowledge graph structure during research.

All types use TypedDict for LangGraph compatibility with
AsyncPostgresSaver checkpointing.
"""

from __future__ import annotations

import uuid
from collections import deque
from datetime import datetime
from typing import Any, Literal, TypedDict


# =============================================================================
# Type Aliases
# =============================================================================

EntityType = Literal[
    "person",
    "organization",
    "concept",
    "event",
    "location",
    "document",
    "claim",
]

ClaimStatus = Literal["supported", "contested", "refuted", "unverified"]


# =============================================================================
# TypedDict Definitions
# =============================================================================


class Entity(TypedDict):
    """
    A node in the knowledge graph representing a real-world entity.

    Entities are extracted from research findings and connected
    through relations to form a semantic network.
    """

    # Unique identifier
    id: str

    # Human-readable name
    name: str

    # Type classification
    entity_type: EntityType

    # Additional properties (flexible schema)
    properties: dict[str, Any]

    # Optional embedding vector for semantic search
    embedding: list[float] | None

    # Citation IDs that mention this entity
    source_ids: list[str]

    # Creation timestamp
    created_at: str


class Relation(TypedDict):
    """
    An edge connecting two entities in the knowledge graph.

    Relations capture semantic relationships discovered during research,
    such as authorship, mentions, support, or contradiction.
    """

    # Unique identifier
    id: str

    # Source entity ID (subject)
    source_entity_id: str

    # Target entity ID (object)
    target_entity_id: str

    # Relationship type (predicate)
    relation_type: str

    # Confidence in this relation (0.0-1.0)
    confidence: float

    # Evidence text supporting this relation
    evidence: str

    # Citation ID where this was found
    source_id: str

    # Creation timestamp
    created_at: str


class Claim(TypedDict):
    """
    A factual claim extracted from research.

    Claims represent assertions that may be supported, contested,
    or refuted by evidence from multiple sources.
    """

    # Unique identifier
    id: str

    # The claim statement
    statement: str

    # Entity IDs that support this claim
    supporting_entity_ids: list[str]

    # Finding/evidence IDs
    evidence_ids: list[str]

    # Confidence score (0.0-1.0)
    confidence: float

    # Verification status
    status: ClaimStatus

    # Creation timestamp
    created_at: str


# =============================================================================
# Knowledge Graph Class
# =============================================================================


class KnowledgeGraph:
    """
    In-memory knowledge graph with query and export capabilities.

    Manages entities, relations, and claims for a research session.
    Provides export formats for frontend visualization and interoperability.

    Example:
        ```python
        graph = KnowledgeGraph()

        # Add entities
        entity = create_entity("OpenAI", "organization")
        graph.add_entity(entity)

        # Add relations
        relation = create_relation(
            source_entity_id=entity["id"],
            target_entity_id=other_entity["id"],
            relation_type="created",
            evidence="OpenAI created ChatGPT",
            source_id="citation_123"
        )
        graph.add_relation(relation)

        # Export for frontend
        cytoscape_data = graph.export_cytoscape()
        ```
    """

    def __init__(self) -> None:
        """Initialize an empty knowledge graph."""
        self._entities: dict[str, Entity] = {}
        self._relations: dict[str, Relation] = {}
        self._claims: dict[str, Claim] = {}

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def add_entity(self, entity: Entity) -> str:
        """
        Add an entity to the graph.

        Args:
            entity: The entity to add

        Returns:
            The entity's ID
        """
        self._entities[entity["id"]] = entity
        return entity["id"]

    def add_relation(self, relation: Relation) -> str:
        """
        Add a relation between entities.

        Args:
            relation: The relation to add

        Returns:
            The relation's ID
        """
        self._relations[relation["id"]] = relation
        return relation["id"]

    def add_claim(self, claim: Claim) -> str:
        """
        Add a claim to the graph.

        Args:
            claim: The claim to add

        Returns:
            The claim's ID
        """
        self._claims[claim["id"]] = claim
        return claim["id"]

    def get_entity(self, entity_id: str) -> Entity | None:
        """Retrieve an entity by ID."""
        return self._entities.get(entity_id)

    def get_relation(self, relation_id: str) -> Relation | None:
        """Retrieve a relation by ID."""
        return self._relations.get(relation_id)

    def get_claim(self, claim_id: str) -> Claim | None:
        """Retrieve a claim by ID."""
        return self._claims.get(claim_id)

    # =========================================================================
    # Query Operations
    # =========================================================================

    def get_related_entities(
        self,
        entity_id: str,
        relation_type: str | None = None,
        direction: Literal["outgoing", "incoming", "both"] = "both",
    ) -> list[Entity]:
        """
        Get entities related to a given entity.

        Args:
            entity_id: The source entity ID
            relation_type: Optional filter by relation type
            direction: Filter by relation direction

        Returns:
            List of related entities
        """
        related_ids: set[str] = set()

        for relation in self._relations.values():
            # Filter by relation type if specified
            if relation_type and relation["relation_type"] != relation_type:
                continue

            # Check outgoing relations
            if direction in ("outgoing", "both"):
                if relation["source_entity_id"] == entity_id:
                    related_ids.add(relation["target_entity_id"])

            # Check incoming relations
            if direction in ("incoming", "both"):
                if relation["target_entity_id"] == entity_id:
                    related_ids.add(relation["source_entity_id"])

        return [
            self._entities[eid]
            for eid in related_ids
            if eid in self._entities
        ]

    def get_entity_claims(self, entity_id: str) -> list[Claim]:
        """
        Get all claims involving an entity.

        Args:
            entity_id: The entity ID to search for

        Returns:
            List of claims referencing this entity
        """
        return [
            claim
            for claim in self._claims.values()
            if entity_id in claim["supporting_entity_ids"]
        ]

    def get_subgraph(self, entity_id: str, depth: int = 2) -> dict[str, Any]:
        """
        Extract a subgraph centered on an entity using BFS.

        Args:
            entity_id: The center entity ID
            depth: Maximum traversal depth

        Returns:
            Dict with entities, relations, and claims in the subgraph
        """
        visited_entities: set[str] = set()
        subgraph_relations: list[Relation] = []

        # BFS traversal
        queue: deque[tuple[str, int]] = deque([(entity_id, 0)])

        while queue:
            current_id, current_depth = queue.popleft()

            if current_id in visited_entities:
                continue

            visited_entities.add(current_id)

            if current_depth >= depth:
                continue

            # Find connected entities
            for relation in self._relations.values():
                if relation["source_entity_id"] == current_id:
                    if relation not in subgraph_relations:
                        subgraph_relations.append(relation)
                    if relation["target_entity_id"] not in visited_entities:
                        queue.append((relation["target_entity_id"], current_depth + 1))

                elif relation["target_entity_id"] == current_id:
                    if relation not in subgraph_relations:
                        subgraph_relations.append(relation)
                    if relation["source_entity_id"] not in visited_entities:
                        queue.append((relation["source_entity_id"], current_depth + 1))

        # Collect entities
        subgraph_entities = [
            self._entities[eid]
            for eid in visited_entities
            if eid in self._entities
        ]

        # Collect claims for these entities
        subgraph_claims = [
            claim
            for claim in self._claims.values()
            if any(eid in visited_entities for eid in claim["supporting_entity_ids"])
        ]

        return {
            "entities": subgraph_entities,
            "relations": subgraph_relations,
            "claims": subgraph_claims,
            "center_entity_id": entity_id,
            "depth": depth,
        }

    def find_entities_by_name(
        self,
        name: str,
        entity_type: EntityType | None = None,
    ) -> list[Entity]:
        """
        Find entities by name (case-insensitive partial match).

        Args:
            name: Name to search for
            entity_type: Optional type filter

        Returns:
            Matching entities
        """
        name_lower = name.lower()
        results = []

        for entity in self._entities.values():
            if name_lower in entity["name"].lower():
                if entity_type is None or entity["entity_type"] == entity_type:
                    results.append(entity)

        return results

    # =========================================================================
    # Export Operations
    # =========================================================================

    def export_cytoscape(self) -> dict[str, Any]:
        """
        Export graph in Cytoscape.js format for frontend visualization.

        Returns:
            Dict with nodes and edges arrays
        """
        nodes = []
        edges = []

        # Export entities as nodes
        for entity in self._entities.values():
            nodes.append({
                "data": {
                    "id": entity["id"],
                    "label": entity["name"],
                    "type": entity["entity_type"],
                    **entity.get("properties", {}),
                },
            })

        # Export relations as edges
        for relation in self._relations.values():
            edges.append({
                "data": {
                    "id": relation["id"],
                    "source": relation["source_entity_id"],
                    "target": relation["target_entity_id"],
                    "label": relation["relation_type"],
                    "confidence": relation["confidence"],
                },
            })

        return {
            "nodes": nodes,
            "edges": edges,
        }

    def export_jsonld(self) -> dict[str, Any]:
        """
        Export graph in JSON-LD format for interoperability.

        Uses schema.org vocabulary where applicable.

        Returns:
            JSON-LD document
        """
        # Map entity types to schema.org types
        type_mapping = {
            "person": "Person",
            "organization": "Organization",
            "concept": "Thing",
            "event": "Event",
            "location": "Place",
            "document": "CreativeWork",
            "claim": "Claim",
        }

        graph_items = []

        # Export entities
        for entity in self._entities.values():
            item = {
                "@id": f"urn:drx:entity:{entity['id']}",
                "@type": type_mapping.get(entity["entity_type"], "Thing"),
                "name": entity["name"],
            }
            # Add properties
            for key, value in entity.get("properties", {}).items():
                if key not in ("@id", "@type", "name"):
                    item[key] = value

            graph_items.append(item)

        # Export claims
        for claim in self._claims.values():
            item = {
                "@id": f"urn:drx:claim:{claim['id']}",
                "@type": "Claim",
                "text": claim["statement"],
                "claimStatus": claim["status"],
                "confidence": claim["confidence"],
            }
            graph_items.append(item)

        return {
            "@context": {
                "@vocab": "https://schema.org/",
                "confidence": {"@type": "xsd:float"},
                "claimStatus": {"@type": "xsd:string"},
            },
            "@graph": graph_items,
        }

    # =========================================================================
    # Merge Operations
    # =========================================================================

    def merge(self, other: KnowledgeGraph) -> None:
        """
        Merge another graph into this one.

        Entities with the same name and type are deduplicated.

        Args:
            other: The graph to merge in
        """
        # Build name+type to ID mapping for deduplication
        existing_key_to_id: dict[str, str] = {}
        for entity in self._entities.values():
            key = f"{entity['name'].lower()}:{entity['entity_type']}"
            existing_key_to_id[key] = entity["id"]

        # Map other IDs to existing IDs for deduplication
        id_mapping: dict[str, str] = {}

        # Merge entities
        for entity in other._entities.values():
            key = f"{entity['name'].lower()}:{entity['entity_type']}"
            if key in existing_key_to_id:
                # Entity exists, map to existing ID
                id_mapping[entity["id"]] = existing_key_to_id[key]
            else:
                # New entity
                self._entities[entity["id"]] = entity
                id_mapping[entity["id"]] = entity["id"]

        # Merge relations with remapped IDs
        for relation in other._relations.values():
            new_relation = dict(relation)
            new_relation["source_entity_id"] = id_mapping.get(
                relation["source_entity_id"],
                relation["source_entity_id"],
            )
            new_relation["target_entity_id"] = id_mapping.get(
                relation["target_entity_id"],
                relation["target_entity_id"],
            )
            self._relations[relation["id"]] = new_relation  # type: ignore

        # Merge claims with remapped entity IDs
        for claim in other._claims.values():
            new_claim = dict(claim)
            new_claim["supporting_entity_ids"] = [
                id_mapping.get(eid, eid)
                for eid in claim["supporting_entity_ids"]
            ]
            self._claims[claim["id"]] = new_claim  # type: ignore

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def entity_count(self) -> int:
        """Number of entities in the graph."""
        return len(self._entities)

    @property
    def relation_count(self) -> int:
        """Number of relations in the graph."""
        return len(self._relations)

    @property
    def claim_count(self) -> int:
        """Number of claims in the graph."""
        return len(self._claims)

    def __repr__(self) -> str:
        return (
            f"KnowledgeGraph(entities={self.entity_count}, "
            f"relations={self.relation_count}, claims={self.claim_count})"
        )


# =============================================================================
# Factory Functions
# =============================================================================


def _generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


def _timestamp_now() -> str:
    """Get current timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def create_entity(
    name: str,
    entity_type: EntityType,
    properties: dict[str, Any] | None = None,
    source_ids: list[str] | None = None,
    embedding: list[float] | None = None,
    entity_id: str | None = None,
) -> Entity:
    """
    Factory function to create an Entity with generated ID.

    Args:
        name: Human-readable entity name
        entity_type: Type classification
        properties: Optional additional properties
        source_ids: Optional citation IDs
        embedding: Optional embedding vector
        entity_id: Optional custom ID (generated if not provided)

    Returns:
        New Entity instance
    """
    return Entity(
        id=entity_id or _generate_id(),
        name=name,
        entity_type=entity_type,
        properties=properties or {},
        embedding=embedding,
        source_ids=source_ids or [],
        created_at=_timestamp_now(),
    )


def create_relation(
    source_entity_id: str,
    target_entity_id: str,
    relation_type: str,
    evidence: str,
    source_id: str,
    confidence: float = 0.8,
    relation_id: str | None = None,
) -> Relation:
    """
    Factory function to create a Relation with generated ID.

    Args:
        source_entity_id: Subject entity ID
        target_entity_id: Object entity ID
        relation_type: Type of relationship
        evidence: Supporting evidence text
        source_id: Citation ID
        confidence: Confidence score (0.0-1.0)
        relation_id: Optional custom ID

    Returns:
        New Relation instance

    Raises:
        ValueError: If confidence is out of range
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")

    return Relation(
        id=relation_id or _generate_id(),
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        relation_type=relation_type,
        confidence=confidence,
        evidence=evidence,
        source_id=source_id,
        created_at=_timestamp_now(),
    )


def create_claim(
    statement: str,
    supporting_entity_ids: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    confidence: float = 0.5,
    status: ClaimStatus = "unverified",
    claim_id: str | None = None,
) -> Claim:
    """
    Factory function to create a Claim with generated ID.

    Args:
        statement: The claim text
        supporting_entity_ids: Entity IDs supporting this claim
        evidence_ids: Finding/evidence IDs
        confidence: Confidence score (0.0-1.0)
        status: Verification status
        claim_id: Optional custom ID

    Returns:
        New Claim instance

    Raises:
        ValueError: If confidence is out of range
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")

    return Claim(
        id=claim_id or _generate_id(),
        statement=statement,
        supporting_entity_ids=supporting_entity_ids or [],
        evidence_ids=evidence_ids or [],
        confidence=confidence,
        status=status,
        created_at=_timestamp_now(),
    )


# =============================================================================
# Exports
# =============================================================================

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
