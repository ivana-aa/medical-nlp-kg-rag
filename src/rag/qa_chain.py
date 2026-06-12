"""Template-based medical QA chain with evidence citations."""

from __future__ import annotations

import re

from src.rag.retriever import MedicalRetriever
from src.rag.splitter import TextChunk


DEFAULT_DISCLAIMER = "风险提示：本回答仅供学习和参考，不能替代医生诊断或治疗建议；如有不适或病情变化，请及时到正规医疗机构就诊。"


class MedicalQAChain:
    """A lightweight RAG QA chain.

    This class intentionally uses a template answer instead of an external LLM
    so the demo can run without API keys. TODO: Replace ``generate_answer`` with
    a local LLM or a hosted model when stronger generation is needed.
    """

    def __init__(
        self,
        retriever: MedicalRetriever,
        disclaimer: str = DEFAULT_DISCLAIMER,
        max_evidence_chars: int = 900,
    ) -> None:
        self.retriever = retriever
        self.disclaimer = disclaimer
        self.max_evidence_chars = max_evidence_chars

    def answer(self, question: str) -> dict[str, object]:
        """Retrieve evidence and generate a cited answer."""
        question = question.strip()
        if not question:
            return {
                "answer": f"请输入具体的医学问题。\n\n{self.disclaimer}",
                "evidence": [],
            }

        results = self.retriever.retrieve(question)
        evidence = self._format_evidence(results)
        answer = self.generate_answer(question, evidence)
        return {"answer": answer, "evidence": evidence}

    def generate_answer(self, question: str, evidence: list[dict[str, object]]) -> str:
        """Create a concise answer from retrieved evidence."""
        if not evidence:
            return (
                "本地知识库中没有检索到足够相关的资料。建议补充更完整、来源可靠的医学文本后重新构建向量库。\n\n"
                f"{self.disclaimer}"
            )

        bullet_lines = []
        for idx, item in enumerate(evidence, start=1):
            snippet = _shorten(str(item["content"]), 180)
            bullet_lines.append(f"{idx}. {snippet} [{idx}]")

        return (
            f"问题：{question}\n\n"
            "根据本地医学知识库检索到的证据，相关信息如下：\n"
            + "\n".join(bullet_lines)
            + "\n\n"
            "综合来看，上述内容可以作为了解疾病表现、常见检查、治疗原则或就医沟通的参考；"
            "但具体病因、诊断和用药需要结合个人病史、体格检查和医生判断。\n\n"
            f"{self.disclaimer}"
        )

    def _format_evidence(self, results: list[dict[str, object]]) -> list[dict[str, object]]:
        evidence: list[dict[str, object]] = []
        used_chars = 0
        for result in results:
            chunk: TextChunk = result["chunk"]  # type: ignore[assignment]
            content = chunk.content.strip()
            if not content:
                continue
            if used_chars >= self.max_evidence_chars:
                break
            clipped = content[: max(0, self.max_evidence_chars - used_chars)]
            evidence.append(
                {
                    "content": clipped,
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    "score": float(result["score"]),
                    "metadata": chunk.metadata,
                }
            )
            used_chars += len(clipped)
        return evidence


def _shorten(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"
