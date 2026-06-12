"""Graph-RAG chain that combines KG triples with existing FAISS retrieval."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from src.kg.kg_loader import MedicalKnowledgeGraph, load_medical_kg
from src.kg.kg_query import format_triples, match_entities, query_entity


GRAPH_RAG_DISCLAIMER = (
    "风险提示：以上内容仅供医学学习和项目演示参考，不能替代医生诊断、治疗建议或用药指导。"
    "如有不适，请及时到正规医疗机构就医。"
)


class MedicalGraphRAG:
    """Template Graph-RAG over a lightweight medical knowledge graph."""

    def __init__(
        self,
        kg: MedicalKnowledgeGraph,
        retriever: object | None = None,
        top_k: int = 4,
        disclaimer: str = GRAPH_RAG_DISCLAIMER,
    ) -> None:
        self.kg = kg
        self.retriever = retriever
        self.top_k = top_k
        self.disclaimer = disclaimer

    @classmethod
    def from_csv(
        cls,
        kg_path: str | Path,
        retriever: object | None = None,
        top_k: int = 4,
    ) -> "MedicalGraphRAG":
        """Create a Graph-RAG chain from a CSV graph."""
        return cls(load_medical_kg(kg_path), retriever=retriever, top_k=top_k)

    def answer(self, question: str) -> dict[str, object]:
        """Run entity matching, graph retrieval, text retrieval, and answer generation."""
        question = question.strip()
        if not question:
            return {
                "answer": f"请输入具体医学问题。\n\n{self.disclaimer}",
                "matched_entities": [],
                "graph_evidence": [],
                "text_evidence": [],
                "disclaimer": self.disclaimer,
            }

        matched_entities = match_entities(question, self.kg)
        graph_triples = []
        for entity in matched_entities:
            graph_triples.extend(query_entity(self.kg, entity))
        graph_triples = _dedupe_triples(graph_triples)

        text_evidence = self._retrieve_text_evidence(question)
        answer = self._generate_answer(question, matched_entities, graph_triples, text_evidence)
        return {
            "answer": answer,
            "matched_entities": matched_entities,
            "graph_evidence": [triple.to_dict() for triple in graph_triples],
            "graph_evidence_lines": format_triples(graph_triples),
            "text_evidence": text_evidence,
            "disclaimer": self.disclaimer,
        }

    def _retrieve_text_evidence(self, question: str) -> list[dict[str, object]]:
        if self.retriever is None:
            return []
        try:
            results = self.retriever.retrieve(question, top_k=self.top_k)  # type: ignore[attr-defined]
        except Exception:
            return []

        evidence: list[dict[str, object]] = []
        for result in results:
            chunk = result.get("chunk")
            if chunk is None:
                continue
            evidence.append(
                {
                    "content": _shorten(str(getattr(chunk, "content", "")), 280),
                    "source": str(getattr(chunk, "source", "")),
                    "score": float(result.get("score", 0.0)),
                    "metadata": getattr(chunk, "metadata", {}),
                }
            )
        return evidence

    def _generate_answer(
        self,
        question: str,
        matched_entities: list[str],
        graph_triples: list[object],
        text_evidence: list[dict[str, object]],
    ) -> str:
        if not graph_triples and not text_evidence:
            return (
                f"问题：{question}\n\n"
                "当前图谱和本地文本知识库没有检索到足够相关的证据。建议补充更完整、来源可靠的医学资料后重新检索。\n\n"
                f"{self.disclaimer}"
            )

        relation_groups: dict[str, list[str]] = defaultdict(list)
        for triple in graph_triples:
            relation = str(getattr(triple, "relation", ""))
            tail = str(getattr(triple, "tail", ""))
            if tail and tail not in relation_groups[relation]:
                relation_groups[relation].append(tail)

        direct_parts: list[str] = []
        subject = "、".join(matched_entities) if matched_entities else "问题中的相关实体"
        if relation_groups.get("常见症状"):
            direct_parts.append(f"{subject}常见相关症状包括{_join_cn(relation_groups['常见症状'])}等")
        if relation_groups.get("推荐检查"):
            direct_parts.append(f"常见相关检查包括{_join_cn(relation_groups['推荐检查'])}等")
        if relation_groups.get("就诊科室"):
            direct_parts.append(f"可参考的就诊科室包括{_join_cn(relation_groups['就诊科室'])}")
        if relation_groups.get("相关药物"):
            direct_parts.append(f"相关药物信息包括{_join_cn(relation_groups['相关药物'])}")
        if relation_groups.get("用药风险"):
            direct_parts.append(f"需要关注的用药风险包括{_join_cn(relation_groups['用药风险'])}")

        if not direct_parts:
            direct_parts.append("根据当前图谱，已找到若干结构化医学关联，可结合文本证据进一步参考")

        graph_lines = format_triples(graph_triples)
        graph_block = "\n".join(f"- {line}" for line in graph_lines) if graph_lines else "- 未检索到图谱证据"
        text_block = (
            "\n".join(
                f"- [{idx}] {_shorten(str(item['content']), 180)}"
                for idx, item in enumerate(text_evidence, start=1)
            )
            if text_evidence
            else "- 未检索到文本证据"
        )

        return (
            f"问题：{question}\n\n"
            "直接回答："
            + "；".join(direct_parts)
            + "。\n\n"
            "图谱结构化证据：\n"
            + graph_block
            + "\n\n文本检索证据：\n"
            + text_block
            + "\n\n"
            + self.disclaimer
        )


def _dedupe_triples(triples: list[object]) -> list[object]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[object] = []
    for triple in triples:
        key = (str(getattr(triple, "head", "")), str(getattr(triple, "relation", "")), str(getattr(triple, "tail", "")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(triple)
    return unique


def _shorten(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _join_cn(items: list[str]) -> str:
    return "、".join(items)
