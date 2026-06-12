"""Prepare demo data for NER and RAG knowledge base.

The demo corpus is manually written medical education text. It is not copied
from patient records and contains no private health information.
"""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


NER_EXAMPLES = [
    (
        "高血压患者出现头痛和胸闷，医生建议进行血压检查，并使用硝苯地平。",
        [
            ("高血压", "DISEASE"),
            ("头痛", "SYMPTOM"),
            ("胸闷", "SYMPTOM"),
            ("血压检查", "EXAM"),
            ("硝苯地平", "DRUG"),
        ],
    ),
    (
        "糖尿病患者可能出现多饮、多尿，医生会关注空腹血糖和糖化血红蛋白。",
        [
            ("糖尿病", "DISEASE"),
            ("多饮", "SYMPTOM"),
            ("多尿", "SYMPTOM"),
            ("空腹血糖", "TEST"),
            ("糖化血红蛋白", "TEST"),
        ],
    ),
    (
        "胃炎常见症状包括上腹痛和恶心，可通过胃镜检查辅助判断。",
        [
            ("胃炎", "DISEASE"),
            ("上腹痛", "SYMPTOM"),
            ("恶心", "SYMPTOM"),
            ("胃镜检查", "EXAM"),
        ],
    ),
    (
        "急性阑尾炎患者可能出现右下腹疼痛，严重时需要阑尾切除术。",
        [
            ("急性阑尾炎", "DISEASE"),
            ("右下腹", "BODY"),
            ("疼痛", "SYMPTOM"),
            ("阑尾切除术", "SURGERY"),
        ],
    ),
    (
        "肺部CT检查有助于观察肺部感染，咳嗽和发热是常见表现。",
        [
            ("肺部CT检查", "EXAM"),
            ("肺部", "BODY"),
            ("肺部感染", "DISEASE"),
            ("咳嗽", "SYMPTOM"),
            ("发热", "SYMPTOM"),
        ],
    ),
    (
        "阿莫西林常用于细菌感染相关疾病，用药前应注意过敏史。",
        [
            ("阿莫西林", "DRUG"),
            ("细菌感染", "DISEASE"),
            ("过敏", "SYMPTOM"),
        ],
    ),
    (
        "冠心病患者如果出现胸痛，需要结合心电图和肌钙蛋白检查。",
        [
            ("冠心病", "DISEASE"),
            ("胸痛", "SYMPTOM"),
            ("心电图", "EXAM"),
            ("肌钙蛋白", "TEST"),
        ],
    ),
    (
        "脑卒中可能表现为肢体无力、言语不清，应尽快到医院评估。",
        [
            ("脑卒中", "DISEASE"),
            ("肢体", "BODY"),
            ("无力", "SYMPTOM"),
            ("言语不清", "SYMPTOM"),
        ],
    ),
    (
        "慢性阻塞性肺疾病患者常有气短，肺功能检查有助于评估病情。",
        [
            ("慢性阻塞性肺疾病", "DISEASE"),
            ("气短", "SYMPTOM"),
            ("肺功能检查", "EXAM"),
        ],
    ),
    (
        "甲状腺结节通常需要结合甲状腺超声和促甲状腺激素结果判断。",
        [
            ("甲状腺结节", "DISEASE"),
            ("甲状腺", "BODY"),
            ("甲状腺超声", "EXAM"),
            ("促甲状腺激素", "TEST"),
        ],
    ),
    (
        "骨折患者可能需要X线检查，部分情况需进行内固定手术。",
        [
            ("骨折", "DISEASE"),
            ("X线检查", "EXAM"),
            ("内固定手术", "SURGERY"),
        ],
    ),
    (
        "贫血常见表现包括乏力和头晕，血红蛋白是重要检验指标。",
        [
            ("贫血", "DISEASE"),
            ("乏力", "SYMPTOM"),
            ("头晕", "SYMPTOM"),
            ("血红蛋白", "TEST"),
        ],
    ),
]

NER_ENTITY_CASES = [
    {
        "disease": "高血压",
        "symptoms": ["头痛", "头晕", "胸闷"],
        "exams": ["血压检查", "动态血压监测"],
        "drugs": ["硝苯地平", "氨氯地平"],
        "tests": ["血压"],
        "bodies": ["心脏"],
        "surgeries": [],
    },
    {
        "disease": "糖尿病",
        "symptoms": ["多饮", "多尿", "体重下降"],
        "exams": ["眼底检查", "尿常规检查"],
        "drugs": ["二甲双胍", "胰岛素"],
        "tests": ["空腹血糖", "糖化血红蛋白"],
        "bodies": ["胰腺"],
        "surgeries": [],
    },
    {
        "disease": "胃炎",
        "symptoms": ["上腹痛", "恶心", "反酸"],
        "exams": ["胃镜检查", "幽门螺杆菌检测"],
        "drugs": ["奥美拉唑", "铝碳酸镁"],
        "tests": ["幽门螺杆菌"],
        "bodies": ["胃部"],
        "surgeries": [],
    },
    {
        "disease": "急性阑尾炎",
        "symptoms": ["右下腹疼痛", "发热", "恶心"],
        "exams": ["腹部超声", "血常规检查"],
        "drugs": ["头孢曲松"],
        "tests": ["白细胞计数", "C反应蛋白"],
        "bodies": ["右下腹", "阑尾"],
        "surgeries": ["阑尾切除术"],
    },
    {
        "disease": "肺部感染",
        "symptoms": ["咳嗽", "咳痰", "发热"],
        "exams": ["肺部CT检查", "胸片检查"],
        "drugs": ["阿莫西林", "左氧氟沙星"],
        "tests": ["白细胞计数", "降钙素原"],
        "bodies": ["肺部"],
        "surgeries": [],
    },
    {
        "disease": "冠心病",
        "symptoms": ["胸痛", "胸闷", "气短"],
        "exams": ["心电图", "冠状动脉造影"],
        "drugs": ["阿司匹林", "硝酸甘油"],
        "tests": ["肌钙蛋白", "血脂"],
        "bodies": ["心脏", "冠状动脉"],
        "surgeries": ["冠状动脉支架植入术"],
    },
    {
        "disease": "脑卒中",
        "symptoms": ["肢体无力", "言语不清", "口角歪斜"],
        "exams": ["头颅CT检查", "头颅MRI检查"],
        "drugs": ["阿替普酶", "阿司匹林"],
        "tests": ["凝血功能", "血糖"],
        "bodies": ["脑部", "肢体"],
        "surgeries": [],
    },
    {
        "disease": "贫血",
        "symptoms": ["乏力", "头晕", "心悸"],
        "exams": ["血常规检查", "骨髓检查"],
        "drugs": ["硫酸亚铁", "叶酸"],
        "tests": ["血红蛋白", "铁蛋白"],
        "bodies": ["血液"],
        "surgeries": [],
    },
    {
        "disease": "甲状腺结节",
        "symptoms": ["颈部肿块", "吞咽不适"],
        "exams": ["甲状腺超声", "细针穿刺检查"],
        "drugs": [],
        "tests": ["促甲状腺激素", "甲状腺功能"],
        "bodies": ["甲状腺", "颈部"],
        "surgeries": ["甲状腺切除术"],
    },
    {
        "disease": "骨折",
        "symptoms": ["疼痛", "肿胀", "活动受限"],
        "exams": ["X线检查", "CT检查"],
        "drugs": ["布洛芬"],
        "tests": [],
        "bodies": ["腕部", "下肢"],
        "surgeries": ["内固定手术"],
    },
]

CMEEE_TYPE_MAP = {
    "dis": "DISEASE",
    "sym": "SYMPTOM",
    "dru": "DRUG",
    "equ": "EXAM",
    "pro": "SURGERY",
    "bod": "BODY",
    "ite": "TEST",
    "mic": "TEST",
    "dep": "EXAM",
}


def label_text(text: str, entities: list[tuple[str, str]]) -> list[str]:
    """Create BIO labels for a sentence by matching demo entity strings."""
    labels = ["O"] * len(text)
    for entity, entity_type in sorted(entities, key=lambda item: len(item[0]), reverse=True):
        start = text.find(entity)
        while start != -1:
            end = start + len(entity)
            if all(label == "O" for label in labels[start:end]):
                labels[start] = f"B-{entity_type}"
                for idx in range(start + 1, end):
                    labels[idx] = f"I-{entity_type}"
            start = text.find(entity, end)
    return labels


def write_bio(path: Path, examples: list[tuple[str, list[tuple[str, str]]]]) -> None:
    """Write examples to a BIO file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for text, entities in examples:
            labels = label_text(text, entities)
            for char, label in zip(text, labels):
                f.write(f"{char} {label}\n")
            f.write("\n")


def first_or_empty(values: list[str]) -> str:
    """Return the first value from a list, or an empty string."""
    return values[0] if values else ""


def build_case_ner_examples(case: dict[str, object]) -> list[tuple[str, list[tuple[str, str]]]]:
    """Build templated examples for one curated medical case."""
    disease = str(case["disease"])
    symptoms = list(case["symptoms"])
    exams = list(case["exams"])
    drugs = list(case["drugs"])
    tests = list(case["tests"])
    bodies = list(case["bodies"])
    surgeries = list(case["surgeries"])

    symptom_a = symptoms[0]
    symptom_b = symptoms[1] if len(symptoms) > 1 else symptoms[0]
    exam_a = first_or_empty(exams)
    exam_b = exams[1] if len(exams) > 1 else exam_a
    drug_a = first_or_empty(drugs)
    test_a = first_or_empty(tests)
    test_b = tests[1] if len(tests) > 1 else test_a
    body_a = first_or_empty(bodies)
    surgery_a = first_or_empty(surgeries)

    examples: list[tuple[str, list[tuple[str, str]]]] = [
        (
            f"{disease}患者常见表现包括{symptom_a}和{symptom_b}，医生会结合{exam_a}进行评估。",
            [(disease, "DISEASE"), (symptom_a, "SYMPTOM"), (symptom_b, "SYMPTOM"), (exam_a, "EXAM")],
        ),
        (
            f"医生询问{disease}患者是否存在{symptom_a}、{symptom_b}等症状，并记录相关病情变化。",
            [(disease, "DISEASE"), (symptom_a, "SYMPTOM"), (symptom_b, "SYMPTOM")],
        ),
        (
            f"随访{disease}时，建议观察{symptom_a}变化，并按需复查{exam_a}。",
            [(disease, "DISEASE"), (symptom_a, "SYMPTOM"), (exam_a, "EXAM")],
        ),
    ]

    if drug_a:
        examples.append(
            (
                f"对于{disease}，临床可能根据情况使用{drug_a}，同时观察{symptom_a}是否缓解。",
                [(disease, "DISEASE"), (drug_a, "DRUG"), (symptom_a, "SYMPTOM")],
            )
        )
    if test_a:
        examples.append(
            (
                f"{test_a}是评估{disease}的重要检验指标，必要时还会复查{test_b}。",
                [(test_a, "TEST"), (disease, "DISEASE"), (test_b, "TEST")],
            )
        )
    if body_a:
        examples.append(
            (
                f"当{body_a}出现{symptom_a}时，需要排查{disease}并完善{exam_b}。",
                [(body_a, "BODY"), (symptom_a, "SYMPTOM"), (disease, "DISEASE"), (exam_b, "EXAM")],
            )
        )
    if surgery_a:
        examples.append(
            (
                f"{disease}病情严重时可能需要{surgery_a}，术前通常要完成{exam_a}。",
                [(disease, "DISEASE"), (surgery_a, "SURGERY"), (exam_a, "EXAM")],
            )
        )

    return examples


def build_demo_ner_splits() -> tuple[
    list[tuple[str, list[tuple[str, str]]]],
    list[tuple[str, list[tuple[str, str]]]],
    list[tuple[str, list[tuple[str, str]]]],
]:
    """Build stratified train/dev/test demo splits.

    The examples are templated but still useful for a beginner project because
    they cover all target entity types and give the model repeated, consistent
    BIO patterns. TODO: replace this corpus with CMeEE for formal experiments.
    """
    train_examples = list(NER_EXAMPLES[:8])
    dev_examples = list(NER_EXAMPLES[8:10])
    test_examples = list(NER_EXAMPLES[10:])

    for case in NER_ENTITY_CASES:
        case_examples = build_case_ner_examples(case)
        train_examples.extend(case_examples[:4])
        dev_examples.append(case_examples[4 % len(case_examples)])
        test_examples.append(case_examples[-1])

    return train_examples, dev_examples, test_examples


def write_labeled_chars(path: Path, items: list[tuple[str, list[str]]]) -> None:
    """Write pre-labeled character sequences to a BIO file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for text, labels in items:
            for char, label in zip(text, labels):
                f.write(f"{char} {label}\n")
            f.write("\n")


def find_cmeee_file(cmeee_dir: Path, split: str) -> Path | None:
    """Find CMeEE split files under common names or extracted folders."""
    split_aliases = {
        "train": ["train", "training"],
        "dev": ["dev", "valid", "validation"],
        "test": ["test"],
    }
    aliases = split_aliases.get(split, [split])

    candidates = [
        cmeee_dir / f"CMeEE_{split}.json",
        cmeee_dir / f"cmeee_{split}.json",
        cmeee_dir / f"CMeEE-V2_{split}.json",
        cmeee_dir / f"CMeEE-V2_{split}_1007.json",
        cmeee_dir / f"{split}.json",
    ]
    for path in candidates:
        if path.exists():
            return path

    for path in sorted(cmeee_dir.rglob("*.json")):
        name = path.name.lower()
        if "example" in name or "pred" in name or "gold" in name:
            continue
        if any(alias in name for alias in aliases) and ("cmeee" in name or name in {f"{alias}.json" for alias in aliases}):
            return path
    return None


def convert_cmeee_file(input_path: Path) -> list[tuple[str, list[str]]]:
    """Convert CBLUE/CMeEE JSON examples to flat BIO labels.

    CMeEE contains nested entities. Plain BIO cannot represent overlapping
    spans, so this beginner project keeps the first non-overlapping span and
    skips later spans that collide. TODO: use span classification or layered
    labels if you need full nested entity support.
    """
    raw_text = input_path.read_text(encoding="utf-8")
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        raw = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    if isinstance(raw, dict):
        raw = raw.get("data") or raw.get("examples") or raw.get("items") or []
    converted: list[tuple[str, list[str]]] = []

    for item in raw:
        text = item.get("text", "")
        labels = ["O"] * len(text)
        entities = item.get("entities", [])
        entities = sorted(entities, key=lambda ent: (ent.get("start_idx", 0), ent.get("end_idx", 0)))
        for ent in entities:
            entity_type = CMEEE_TYPE_MAP.get(ent.get("type"))
            if entity_type is None:
                continue
            start = int(ent.get("start_idx", -1))
            end = int(ent.get("end_idx", -1)) + 1
            if start < 0 or end > len(text) or start >= end:
                continue
            if any(label != "O" for label in labels[start:end]):
                continue
            labels[start] = f"B-{entity_type}"
            for idx in range(start + 1, end):
                labels[idx] = f"I-{entity_type}"
        converted.append((text, labels))

    return converted


def prepare_cmeee_data(cmeee_dir: Path) -> bool:
    """Convert downloaded CMeEE files when the user provides a raw directory."""
    output_dir = ROOT / "data" / "processed" / "ner_demo"
    converted_splits: dict[str, list[tuple[str, list[str]]]] = {}
    for split, output_name in [("train", "train.bio"), ("dev", "dev.bio"), ("test", "test.bio")]:
        input_path = find_cmeee_file(cmeee_dir, split)
        if input_path is None:
            print(f"Skip {split}: no CMeEE JSON file found in {cmeee_dir}")
            continue
        items = convert_cmeee_file(input_path)
        if split == "test" and not any(any(label != "O" for label in labels) for _, labels in items):
            print(f"Skip {split}: {input_path} has no entity labels, using a labeled holdout split instead.")
            continue
        converted_splits[split] = items
        write_labeled_chars(output_dir / output_name, items)
        print(f"Converted {input_path} -> {output_dir / output_name} ({len(items)} examples)")

    if "test" not in converted_splits:
        source_split = "dev" if "dev" in converted_splits and len(converted_splits["dev"]) >= 2 else "train"
        source_items = list(converted_splits.get(source_split, []))
        if len(source_items) >= 2:
            cut = max(1, int(len(source_items) * 0.5)) if source_split == "dev" else max(1, int(len(source_items) * 0.9))
            kept_items = source_items[:cut]
            test_items = source_items[cut:]
            if not test_items:
                test_items = kept_items[-1:]
                kept_items = kept_items[:-1]
            converted_splits[source_split] = kept_items
            converted_splits["test"] = test_items
            write_labeled_chars(output_dir / f"{source_split}.bio", kept_items)
            write_labeled_chars(output_dir / "test.bio", test_items)
            print(
                f"Created local labeled test split from CMeEE {source_split}: "
                f"{len(kept_items)} kept, {len(test_items)} test examples"
            )

    return bool(converted_splits)


def prepare_ner_demo() -> None:
    """Create train/dev/test BIO files."""
    output_dir = ROOT / "data" / "processed" / "ner_demo"
    train_dir = ROOT / "data" / "processed" / "ner_train"
    train_examples, dev_examples, test_examples = build_demo_ner_splits()

    for target_dir in [output_dir, train_dir]:
        write_bio(target_dir / "train.bio", train_examples)
        write_bio(target_dir / "dev.bio", dev_examples)
        write_bio(target_dir / "test.bio", test_examples)

    print(f"NER demo data written to: {output_dir}")
    print(f"Default training BIO files written to: {train_dir}")
    print(f"train/dev/test = {len(train_examples)}/{len(dev_examples)}/{len(test_examples)}")


def prepare_knowledge_base_demo() -> None:
    """Create small local knowledge base files for RAG."""
    kb_dir = ROOT / "data" / "knowledge_base"
    kb_dir.mkdir(parents=True, exist_ok=True)

    (kb_dir / "demo_medical_knowledge.md").write_text(
        """# 常见疾病科普知识库

## 高血压
高血压是以动脉血压持续升高为主要特征的慢性疾病。常见风险因素包括高盐饮食、超重、缺乏运动、长期精神紧张和家族史。部分患者可能没有明显症状，也可能出现头痛、头晕、胸闷等表现。管理高血压通常包括规律监测血压、改善生活方式、控制体重、限制钠盐摄入、遵医嘱用药和定期复诊。

## 糖尿病
糖尿病是一组以血糖升高为特征的代谢性疾病。典型表现包括多饮、多尿、多食和体重下降，但早期也可能没有明显症状。常用检查包括空腹血糖、餐后血糖和糖化血红蛋白。日常管理强调饮食控制、规律运动、血糖监测、药物治疗和并发症筛查。

## 普通感冒
普通感冒多由病毒感染引起，常见表现包括鼻塞、流涕、咽痛、咳嗽和低热。多数患者可以通过休息、补充水分和对症处理缓解。若出现持续高热、呼吸困难、胸痛、意识异常或基础疾病加重，应及时就医。

## 胃炎
胃炎可表现为上腹痛、腹胀、恶心、反酸等症状。诱因可能包括幽门螺杆菌感染、药物刺激、饮酒和不规律饮食。医生可能根据症状、胃镜检查、幽门螺杆菌检测等进行评估。治疗通常包括去除诱因、规律饮食和必要时使用抑酸药物。
""",
        encoding="utf-8",
    )

    faq_items = [
        {
            "title": "发热什么时候需要就医",
            "content": "成人发热如果持续不退、体温很高、伴有呼吸困难、胸痛、意识改变、皮疹、严重脱水，或患者存在免疫功能低下、严重基础病，应尽快就医。",
        },
        {
            "title": "胸痛如何处理",
            "content": "胸痛可能与心脏、肺部、消化系统或肌肉骨骼问题有关。若胸痛压榨样、持续不缓解，伴出汗、气短、放射至左肩或下颌，应立即拨打急救电话。",
        },
        {
            "title": "抗生素使用原则",
            "content": "抗生素主要用于细菌感染，不适合自行用于普通病毒性感冒。是否需要抗生素、选择哪种药物和疗程，应由医生根据病情判断。",
        },
    ]
    (kb_dir / "medical_faq.json").write_text(
        json.dumps(faq_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (kb_dir / "lab_tests.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "meaning", "common_use"])
        writer.writeheader()
        writer.writerow(
            {
                "name": "血红蛋白",
                "meaning": "反映血液携氧能力的重要指标，降低时可提示贫血等情况。",
                "common_use": "用于贫血筛查、出血评估和慢性病随访。",
            }
        )
        writer.writerow(
            {
                "name": "糖化血红蛋白",
                "meaning": "反映近2到3个月平均血糖水平。",
                "common_use": "用于糖尿病诊断辅助和长期血糖控制评估。",
            }
        )
        writer.writerow(
            {
                "name": "肌钙蛋白",
                "meaning": "心肌损伤相关指标，升高时需要结合症状和心电图判断。",
                "common_use": "常用于急性胸痛和疑似心肌梗死的评估。",
            }
        )

    print(f"RAG demo knowledge base written to: {kb_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare demo NER data and local RAG knowledge base.")
    parser.add_argument(
        "--cmeee-dir",
        type=Path,
        default=None,
        help="Optional path containing downloaded CMeEE_train/dev/test JSON files.",
    )
    args = parser.parse_args()

    if args.cmeee_dir:
        converted = prepare_cmeee_data(args.cmeee_dir)
        if not converted:
            print("No CMeEE files converted; falling back to tiny demo NER data.")
            prepare_ner_demo()
    else:
        prepare_ner_demo()
    prepare_knowledge_base_demo()


if __name__ == "__main__":
    main()
