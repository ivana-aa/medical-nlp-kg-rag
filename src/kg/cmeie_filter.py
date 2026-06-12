"""Filter converted CMeIE triples into a reviewed KG subset."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.kg.kg_loader import KnowledgeTriple, REQUIRED_COLUMNS, load_medical_kg


DEFAULT_RELATION_MAPPING = {
    "临床表现": "常见症状",
    "影像学检查": "推荐检查",
    "实验室检查": "推荐检查",
    "辅助检查": "推荐检查",
    "内窥镜检查": "推荐检查",
    "组织学检查": "推荐检查",
    "病理检查": "推荐检查",
    "药物治疗": "相关药物",
    "就诊科室": "就诊科室",
    "并发症": "并发症",
    "高危因素": "风险因素",
    "风险评估因素": "风险因素",
    "病因": "病因",
    "发病机制": "发病机制",
    "发病部位": "发病部位",
    "传播途径": "传播途径",
    "预防": "预防",
    "同义词": "同义词",
}

_MARKUP_RE = re.compile(r"https?://|www\.|[\[\]\(\)]")


@dataclass(frozen=True)
class CMeIEFilterConfig:
    """Configuration for building a conservative CMeIE KG subset."""

    relation_mapping: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_RELATION_MAPPING))
    max_entity_chars: int = 40
    max_triples: int | None = None
    max_per_relation: int | None = 2000
    max_per_head: int | None = 30
    reviewed_source_suffix: str = "reviewed"


@dataclass(frozen=True)
class CMeIEFilterStats:
    """Counters produced by one CMeIE KG filtering run."""

    input_triples: int
    kept_triples: int
    dropped_unmapped_relation: int
    dropped_invalid_text: int
    dropped_duplicates: int
    dropped_by_limits: int
    relation_counts: dict[str, int]
    output_path: str = ""
    report_path: str = ""


@dataclass(frozen=True)
class CMeIEFilterResult:
    """Filtered triples plus their statistics."""

    triples: list[KnowledgeTriple]
    stats: CMeIEFilterStats


def filter_cmeie_triples(
    triples: list[KnowledgeTriple],
    config: CMeIEFilterConfig | None = None,
) -> CMeIEFilterResult:
    """Map and filter CMeIE triples into a smaller review-friendly subgraph."""
    config = config or CMeIEFilterConfig()
    output: list[KnowledgeTriple] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    relation_counts: Counter[str] = Counter()
    head_counts: Counter[str] = Counter()

    dropped_unmapped = 0
    dropped_invalid = 0
    dropped_duplicates = 0
    dropped_by_limits = 0

    for triple in triples:
        relation = config.relation_mapping.get(triple.relation)
        if not relation:
            dropped_unmapped += 1
            continue

        head = _clean_text(triple.head)
        tail = _clean_text(triple.tail)
        if not _valid_entity(head, config.max_entity_chars) or not _valid_entity(tail, config.max_entity_chars):
            dropped_invalid += 1
            continue

        key = (head, relation, tail, triple.head_type.strip(), triple.tail_type.strip())
        if key in seen:
            dropped_duplicates += 1
            continue

        if _limit_reached(relation_counts[relation], config.max_per_relation):
            dropped_by_limits += 1
            continue
        if _limit_reached(head_counts[head], config.max_per_head):
            dropped_by_limits += 1
            continue
        if _limit_reached(len(output), config.max_triples):
            dropped_by_limits += 1
            continue

        seen.add(key)
        relation_counts[relation] += 1
        head_counts[head] += 1
        output.append(
            KnowledgeTriple(
                head=head,
                relation=relation,
                tail=tail,
                head_type=triple.head_type.strip(),
                tail_type=triple.tail_type.strip(),
                source=_reviewed_source(triple.source, config.reviewed_source_suffix),
            )
        )

    stats = CMeIEFilterStats(
        input_triples=len(triples),
        kept_triples=len(output),
        dropped_unmapped_relation=dropped_unmapped,
        dropped_invalid_text=dropped_invalid,
        dropped_duplicates=dropped_duplicates,
        dropped_by_limits=dropped_by_limits,
        relation_counts=dict(sorted(relation_counts.items(), key=lambda item: (-item[1], item[0]))),
    )
    return CMeIEFilterResult(triples=output, stats=stats)


def build_review_kg(
    input_path: str | Path,
    output_path: str | Path,
    config: CMeIEFilterConfig | None = None,
    report_path: str | Path | None = None,
) -> CMeIEFilterStats:
    """Build a review-oriented CMeIE KG CSV from a converted CMeIE KG CSV."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    result = filter_cmeie_triples(load_medical_kg(input_path).triples, config)
    _write_kg_csv(output_path, result.triples)

    report_value = ""
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_value = str(report_path)
        report_path.write_text(
            json.dumps(
                {
                    **asdict(result.stats),
                    "output_path": str(output_path),
                    "report_path": report_value,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return CMeIEFilterStats(
        **{
            **asdict(result.stats),
            "output_path": str(output_path),
            "report_path": report_value,
        }
    )


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _valid_entity(value: str, max_chars: int) -> bool:
    if not value or len(value) > max_chars:
        return False
    return _MARKUP_RE.search(value) is None


def _limit_reached(current_count: int, limit: int | None) -> bool:
    return limit is not None and current_count >= limit


def _reviewed_source(source: str, suffix: str) -> str:
    source = source.strip()
    if not suffix:
        return source
    return f"{source}|{suffix}" if source else suffix


def _write_kg_csv(path: Path, triples: list[KnowledgeTriple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for triple in triples:
            writer.writerow(triple.to_dict())
