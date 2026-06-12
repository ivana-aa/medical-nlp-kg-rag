"""NER-assisted candidate relation extraction for the medical KG demo."""

from __future__ import annotations

from dataclasses import asdict, dataclass


ENTITY_TYPE_MAPPING = {
    "DISEASE": "疾病",
    "SYMPTOM": "症状",
    "DRUG": "药物",
    "EXAM": "检查",
    "TEST": "检查",
    "SURGERY": "手术",
    "BODY": "身体部位",
    "DEPARTMENT": "科室",
    "疾病": "疾病",
    "症状": "症状",
    "药物": "药物",
    "检查": "检查",
    "检验指标": "检查",
    "科室": "科室",
}

DEPARTMENT_TERMS = ["呼吸内科", "心血管内科", "内分泌科", "消化内科", "神经内科", "急诊科"]

SYMPTOM_TRIGGERS = ["表现为", "症状", "伴有", "出现", "常见"]
EXAM_TRIGGERS = ["检查", "建议", "可行", "推荐"]
DEPARTMENT_TRIGGERS = ["就诊", "挂号", "科室", "建议"]


@dataclass(frozen=True)
class CandidateRelation:
    """A candidate relation extracted from one sentence."""

    head: str
    relation: str
    tail: str
    head_type: str
    tail_type: str
    evidence: str
    rule: str

    def to_dict(self) -> dict[str, str]:
        """Return a table-friendly dictionary."""
        return asdict(self)


def normalize_entity_type(entity_type: str) -> str:
    """Map CMeEE/NER labels to KG labels."""
    return ENTITY_TYPE_MAPPING.get(entity_type, entity_type)


def enrich_department_entities(text: str, entities: list[dict[str, object]]) -> list[dict[str, object]]:
    """Add simple department matches that the NER model may not cover."""
    enriched = list(entities)
    occupied = {
        idx
        for entity in enriched
        for idx in range(int(entity.get("start", 0)), int(entity.get("end", 0)))
    }
    for term in sorted(DEPARTMENT_TERMS, key=len, reverse=True):
        start = text.find(term)
        while start != -1:
            end = start + len(term)
            if not any(idx in occupied for idx in range(start, end)):
                enriched.append({"text": term, "type": "DEPARTMENT", "start": start, "end": end})
                occupied.update(range(start, end))
            start = text.find(term, end)
    return sorted(enriched, key=lambda item: int(item.get("start", 0)))


def build_candidate_relations(text: str, entities: list[dict[str, object]]) -> list[CandidateRelation]:
    """Generate candidate KG relations from NER entities using transparent rules.

    These candidates are for review and display only. They are not written back
    to the main CSV graph by default, which prevents demo noise from polluting
    curated graph triples.
    """
    entities = enrich_department_entities(text, entities)
    typed_entities: dict[str, list[str]] = {"疾病": [], "症状": [], "检查": [], "科室": []}
    for entity in entities:
        name = str(entity.get("text", "")).strip()
        entity_type = normalize_entity_type(str(entity.get("type", "")))
        if name and entity_type in typed_entities and name not in typed_entities[entity_type]:
            typed_entities[entity_type].append(name)

    candidates: list[CandidateRelation] = []
    if _contains_any(text, SYMPTOM_TRIGGERS):
        candidates.extend(
            CandidateRelation(disease, "可能症状", symptom, "疾病", "症状", text, "疾病+症状+症状触发词")
            for disease in typed_entities["疾病"]
            for symptom in typed_entities["症状"]
        )

    if _contains_any(text, EXAM_TRIGGERS):
        candidates.extend(
            CandidateRelation(disease, "可能检查", exam, "疾病", "检查", text, "疾病+检查+检查触发词")
            for disease in typed_entities["疾病"]
            for exam in typed_entities["检查"]
        )

    if _contains_any(text, DEPARTMENT_TRIGGERS):
        candidates.extend(
            CandidateRelation(disease, "可能科室", department, "疾病", "科室", text, "疾病+科室+就诊触发词")
            for disease in typed_entities["疾病"]
            for department in typed_entities["科室"]
        )

    return _dedupe_candidates(candidates)


def _contains_any(text: str, triggers: list[str]) -> bool:
    return any(trigger in text for trigger in triggers)


def _dedupe_candidates(candidates: list[CandidateRelation]) -> list[CandidateRelation]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[CandidateRelation] = []
    for candidate in candidates:
        key = (candidate.head, candidate.relation, candidate.tail)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique
