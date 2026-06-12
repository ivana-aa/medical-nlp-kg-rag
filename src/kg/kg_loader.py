"""Load a lightweight medical knowledge graph from CSV."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import pandas as pd


REQUIRED_COLUMNS = ["head", "relation", "tail", "head_type", "tail_type", "source"]


@dataclass(frozen=True)
class KnowledgeTriple:
    """One medical knowledge graph edge."""

    head: str
    relation: str
    tail: str
    head_type: str
    tail_type: str
    source: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON/table friendly representation."""
        return asdict(self)


class MedicalKnowledgeGraph:
    """Small CSV-backed medical knowledge graph.

    NetworkX is used for local graph operations only. This keeps the demo
    Windows-friendly and avoids requiring a Neo4j service.
    """

    def __init__(self, triples: list[KnowledgeTriple]) -> None:
        self.triples = triples
        self.graph = nx.MultiDiGraph()
        for triple in triples:
            self.graph.add_node(triple.head, entity_type=triple.head_type)
            self.graph.add_node(triple.tail, entity_type=triple.tail_type)
            self.graph.add_edge(
                triple.head,
                triple.tail,
                relation=triple.relation,
                source=triple.source,
            )

    @classmethod
    def from_csv(cls, path: str | Path) -> "MedicalKnowledgeGraph":
        """Load graph triples from a CSV file."""
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Medical KG CSV not found: {csv_path}")
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"Medical KG CSV missing columns: {missing}")

        triples = [
            KnowledgeTriple(
                head=str(row["head"]).strip(),
                relation=str(row["relation"]).strip(),
                tail=str(row["tail"]).strip(),
                head_type=str(row["head_type"]).strip(),
                tail_type=str(row["tail_type"]).strip(),
                source=str(row["source"]).strip(),
            )
            for _, row in df.iterrows()
            if str(row["head"]).strip() and str(row["relation"]).strip() and str(row["tail"]).strip()
        ]
        return cls(triples)

    def entity_count(self) -> int:
        """Return the number of graph nodes."""
        return self.graph.number_of_nodes()

    def relation_count(self) -> int:
        """Return the number of graph edges."""
        return self.graph.number_of_edges()

    def relation_type_counts(self) -> dict[str, int]:
        """Count triples by relation type."""
        counts: dict[str, int] = {}
        for triple in self.triples:
            counts[triple.relation] = counts.get(triple.relation, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def entities(self) -> list[str]:
        """Return sorted entity names."""
        return sorted(str(node) for node in self.graph.nodes)


def load_medical_kg(path: str | Path) -> MedicalKnowledgeGraph:
    """Convenience loader for the CSV-backed graph."""
    return MedicalKnowledgeGraph.from_csv(path)
