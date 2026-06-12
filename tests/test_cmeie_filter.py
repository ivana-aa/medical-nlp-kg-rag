import tempfile
import unittest
from pathlib import Path

from src.kg.cmeie_filter import CMeIEFilterConfig, build_review_kg, filter_cmeie_triples
from src.kg.kg_loader import KnowledgeTriple, load_medical_kg


def triple(
    head: str,
    relation: str,
    tail: str,
    head_type: str = "疾病",
    tail_type: str = "症状",
    source: str = "CMeIE_train",
) -> KnowledgeTriple:
    return KnowledgeTriple(
        head=head,
        relation=relation,
        tail=tail,
        head_type=head_type,
        tail_type=tail_type,
        source=source,
    )


class CMeIEFilterTests(unittest.TestCase):
    def test_maps_selected_cmeie_relations_to_project_relations(self):
        triples = [
            triple("肺炎", "临床表现", "发热", tail_type="症状"),
            triple("肺炎", "影像学检查", "胸部CT", tail_type="检查"),
            triple("高血压", "药物治疗", "硝苯地平", tail_type="药物"),
        ]

        result = filter_cmeie_triples(triples, CMeIEFilterConfig())

        self.assertEqual(
            [item.relation for item in result.triples],
            ["常见症状", "推荐检查", "相关药物"],
        )
        self.assertEqual(result.stats.input_triples, 3)
        self.assertEqual(result.stats.kept_triples, 3)

    def test_filters_unmapped_empty_markdown_and_overlong_entities(self):
        triples = [
            triple("肺炎", "临床表现", "发热"),
            triple("肺炎", "不展示关系", "无效"),
            triple("", "临床表现", "发热"),
            triple("肺炎", "临床表现", "[链接](http://example.com)"),
            triple("很长" * 30, "临床表现", "发热"),
        ]

        result = filter_cmeie_triples(triples, CMeIEFilterConfig(max_entity_chars=20))

        self.assertEqual([item.to_dict() for item in result.triples], [
            {
                "head": "肺炎",
                "relation": "常见症状",
                "tail": "发热",
                "head_type": "疾病",
                "tail_type": "症状",
                "source": "CMeIE_train|reviewed",
            }
        ])
        self.assertEqual(result.stats.dropped_unmapped_relation, 1)
        self.assertEqual(result.stats.dropped_invalid_text, 3)

    def test_deduplicates_and_applies_relation_and_head_limits(self):
        triples = [
            triple("肺炎", "临床表现", "发热"),
            triple("肺炎", "临床表现", "发热"),
            triple("肺炎", "临床表现", "咳嗽"),
            triple("肺炎", "临床表现", "咳痰"),
            triple("糖尿病", "临床表现", "多饮"),
        ]
        config = CMeIEFilterConfig(max_per_relation=3, max_per_head=2)

        result = filter_cmeie_triples(triples, config)

        self.assertEqual(
            [(item.head, item.relation, item.tail) for item in result.triples],
            [
                ("肺炎", "常见症状", "发热"),
                ("肺炎", "常见症状", "咳嗽"),
                ("糖尿病", "常见症状", "多饮"),
            ],
        )
        self.assertEqual(result.stats.dropped_duplicates, 1)
        self.assertEqual(result.stats.dropped_by_limits, 1)

    def test_build_review_kg_writes_loadable_csv_and_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "cmeie.csv"
            output_path = root / "review.csv"
            report_path = root / "report.json"
            input_path.write_text(
                "head,relation,tail,head_type,tail_type,source\n"
                "肺炎,临床表现,发热,疾病,症状,CMeIE_train\n"
                "肺炎,影像学检查,胸部CT,疾病,检查,CMeIE_train\n",
                encoding="utf-8",
            )

            stats = build_review_kg(input_path, output_path, report_path=report_path)
            kg = load_medical_kg(output_path)
            report_exists = report_path.exists()

        self.assertEqual(stats.kept_triples, 2)
        self.assertEqual(kg.relation_count(), 2)
        self.assertTrue(report_exists)


if __name__ == "__main__":
    unittest.main()
