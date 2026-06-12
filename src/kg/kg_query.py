"""Query helpers for the medical knowledge graph."""

from __future__ import annotations

from src.kg.kg_loader import KnowledgeTriple, MedicalKnowledgeGraph


def query_entity(
    kg: MedicalKnowledgeGraph,
    entity_name: str,
    relation_type: str | None = None,
    include_incoming: bool = True,
) -> list[KnowledgeTriple]:
    """Find triples connected to an entity."""
    entity_name = entity_name.strip()
    if not entity_name:
        return []

    triples: list[KnowledgeTriple] = []
    for triple in kg.triples:
        relation_ok = relation_type in {None, "", "全部", triple.relation}
        if not relation_ok:
            continue
        if triple.head == entity_name or (include_incoming and triple.tail == entity_name):
            triples.append(triple)
    return triples


def filter_by_relation(kg: MedicalKnowledgeGraph, relation_type: str) -> list[KnowledgeTriple]:
    """Return triples of one relation type."""
    relation_type = relation_type.strip()
    if not relation_type or relation_type == "全部":
        return kg.triples
    return [triple for triple in kg.triples if triple.relation == relation_type]


def match_entities(text: str, kg: MedicalKnowledgeGraph, max_entities: int = 5) -> list[str]:
    """Match graph entities mentioned in text by longest exact substring."""
    text = text.strip()
    if not text:
        return []

    matches: list[str] = []
    occupied: set[int] = set()
    for entity in sorted(kg.entities(), key=len, reverse=True):
        start = text.find(entity)
        if start == -1:
            continue
        end = start + len(entity)
        if any(idx in occupied for idx in range(start, end)):
            continue
        matches.append(entity)
        occupied.update(range(start, end))
        if len(matches) >= max_entities:
            break
    return matches


def triples_to_records(triples: list[KnowledgeTriple]) -> list[dict[str, str]]:
    """Convert triples to table-ready records."""
    return [triple.to_dict() for triple in triples]


def format_triples(triples: list[KnowledgeTriple]) -> list[str]:
    """Format triples as readable evidence lines."""
    return [f"{triple.head} - {triple.relation} - {triple.tail}" for triple in triples]
