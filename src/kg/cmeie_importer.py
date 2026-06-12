"""Import CMeIE relation extraction data into the CSV KG format."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.kg.kg_loader import KnowledgeTriple, REQUIRED_COLUMNS


JSON_SUFFIXES = {".json", ".jsonl"}


@dataclass(frozen=True)
class CMeIEImportStats:
    """Summary of one CMeIE-to-KG conversion."""

    records: int
    spo_items: int
    valid_triples: int
    written_triples: int
    output_path: str


def read_cmeie_records(path: str | Path) -> list[dict[str, Any]]:
    """Read CMeIE records from one JSON/JSONL file or a directory."""
    return [record for record, _source in _iter_records_with_source(Path(path))]


def iter_spo_triples(record: dict[str, Any], source: str) -> Iterable[KnowledgeTriple]:
    """Convert one CMeIE record's spo_list into KG triples.

    CMeIE stores objects as dictionaries. The common ``@value`` entry maps
    directly to the predicate. Other object roles are preserved by suffixing the
    relation with ``:<role>`` so information is not silently collapsed.
    """
    for spo in record.get("spo_list", []) or []:
        if not isinstance(spo, dict):
            continue
        subject = _clean_text(spo.get("subject"))
        predicate = _clean_text(spo.get("predicate"))
        subject_type = _clean_text(spo.get("subject_type"))
        object_type = spo.get("object_type")
        if not subject or not predicate:
            continue

        for role, value in _iter_object_items(spo.get("object")):
            tail = _clean_text(value)
            if not tail:
                continue
            relation = predicate if role in {"", "@value"} else f"{predicate}:{role}"
            yield KnowledgeTriple(
                head=subject,
                relation=relation,
                tail=tail,
                head_type=subject_type,
                tail_type=_lookup_object_type(object_type, role),
                source=source,
            )


def convert_cmeie_to_kg(
    input_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
) -> CMeIEImportStats:
    """Convert CMeIE JSON/JSONL data into a deduplicated KG CSV file."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    records_seen = 0
    spo_items = 0
    valid_triples = 0
    deduped: dict[tuple[str, str, str, str, str], KnowledgeTriple] = {}

    for record, source in _iter_records_with_source(input_path):
        if limit is not None and records_seen >= limit:
            break
        records_seen += 1
        spo_items += len(record.get("spo_list", []) or [])
        for triple in iter_spo_triples(record, source=source):
            valid_triples += 1
            key = (triple.head, triple.relation, triple.tail, triple.head_type, triple.tail_type)
            deduped.setdefault(key, triple)

    triples = list(deduped.values())
    _write_kg_csv(output_path, triples)
    return CMeIEImportStats(
        records=records_seen,
        spo_items=spo_items,
        valid_triples=valid_triples,
        written_triples=len(triples),
        output_path=str(output_path),
    )


def _iter_records_with_source(path: Path) -> Iterable[tuple[dict[str, Any], str]]:
    if not path.exists():
        raise FileNotFoundError(f"CMeIE input path not found: {path}")

    files = [path] if path.is_file() else sorted(
        file for file in path.rglob("*") if file.is_file() and file.suffix.lower() in JSON_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No JSON/JSONL files found under: {path}")

    for file_path in files:
        source = file_path.stem
        for record in _read_records_from_file(file_path):
            yield record, source


def _read_records_from_file(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return []

    if path.suffix.lower() == ".jsonl":
        return [_ensure_record(json.loads(line)) for line in text.splitlines() if line.strip()]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [_ensure_record(json.loads(line)) for line in text.splitlines() if line.strip()]

    if isinstance(parsed, list):
        return [_ensure_record(item) for item in parsed]
    if isinstance(parsed, dict):
        return [_ensure_record(parsed)]
    raise ValueError(f"Unsupported CMeIE JSON root in {path}: {type(parsed).__name__}")


def _ensure_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"CMeIE record must be a JSON object, got {type(value).__name__}")
    return value


def _iter_object_items(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for role, role_value in value.items():
            for scalar in _iter_scalar_values(role_value):
                yield str(role), scalar
        return

    for scalar in _iter_scalar_values(value):
        yield "", scalar


def _iter_scalar_values(value: Any) -> Iterable[Any]:
    if isinstance(value, list):
        for item in value:
            yield from _iter_scalar_values(item)
        return
    if isinstance(value, dict):
        if "@value" in value:
            yield from _iter_scalar_values(value["@value"])
        else:
            yield json.dumps(value, ensure_ascii=False, sort_keys=True)
        return
    yield value


def _lookup_object_type(object_type: Any, role: str) -> str:
    if isinstance(object_type, dict):
        return _clean_text(object_type.get(role) or object_type.get("@value"))
    return _clean_text(object_type)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _write_kg_csv(path: Path, triples: list[KnowledgeTriple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for triple in triples:
            writer.writerow(triple.to_dict())
