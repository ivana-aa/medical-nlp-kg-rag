import json
import tempfile
import unittest
from pathlib import Path

from src.kg.cmeie_importer import convert_cmeie_to_kg, iter_spo_triples, read_cmeie_records
from src.kg.kg_loader import load_medical_kg


class CMeIEImporterTests(unittest.TestCase):
    def test_iter_spo_triples_converts_standard_value_object(self):
        record = {
            "text": "慢性胰腺炎可以采用外照射治疗。",
            "spo_list": [
                {
                    "predicate": "放射治疗",
                    "subject": "慢性胰腺炎",
                    "subject_type": "疾病",
                    "object": {"@value": "外照射"},
                    "object_type": {"@value": "其他治疗"},
                }
            ],
        }

        triples = list(iter_spo_triples(record, source="CMeIE:test"))

        self.assertEqual(len(triples), 1)
        self.assertEqual(triples[0].to_dict(), {
            "head": "慢性胰腺炎",
            "relation": "放射治疗",
            "tail": "外照射",
            "head_type": "疾病",
            "tail_type": "其他治疗",
            "source": "CMeIE:test",
        })

    def test_iter_spo_triples_preserves_complex_object_roles(self):
        record = {
            "spo_list": [
                {
                    "predicate": "临床表现",
                    "subject": "肺炎",
                    "subject_type": "疾病",
                    "object": {"@value": "发热", "部位": "肺部"},
                    "object_type": {"@value": "症状", "部位": "身体部位"},
                }
            ],
        }

        triples = list(iter_spo_triples(record, source="CMeIE:train"))

        self.assertEqual(
            [triple.to_dict() for triple in triples],
            [
                {
                    "head": "肺炎",
                    "relation": "临床表现",
                    "tail": "发热",
                    "head_type": "疾病",
                    "tail_type": "症状",
                    "source": "CMeIE:train",
                },
                {
                    "head": "肺炎",
                    "relation": "临床表现:部位",
                    "tail": "肺部",
                    "head_type": "疾病",
                    "tail_type": "身体部位",
                    "source": "CMeIE:train",
                },
            ],
        )

    def test_read_cmeie_records_accepts_json_array_and_jsonl_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "CMeIE_train.json").write_text(
                json.dumps([{"text": "a", "spo_list": []}], ensure_ascii=False),
                encoding="utf-8",
            )
            (root / "CMeIE_dev.jsonl").write_text(
                json.dumps({"text": "b", "spo_list": []}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            records = read_cmeie_records(root)

        self.assertEqual([record["text"] for record in records], ["b", "a"])

    def test_convert_cmeie_to_kg_writes_deduplicated_loadable_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "CMeIE_train.jsonl"
            output_path = root / "medical_kg_cmeie.csv"
            sample = {
                "spo_list": [
                    {
                        "predicate": "药物治疗",
                        "subject": "高血压",
                        "subject_type": "疾病",
                        "object": {"@value": "降压药"},
                        "object_type": {"@value": "药物"},
                    },
                    {
                        "predicate": "药物治疗",
                        "subject": "高血压",
                        "subject_type": "疾病",
                        "object": {"@value": "降压药"},
                        "object_type": {"@value": "药物"},
                    },
                    {
                        "predicate": "",
                        "subject": "高血压",
                        "subject_type": "疾病",
                        "object": {"@value": "无效关系"},
                        "object_type": {"@value": "药物"},
                    },
                ]
            }
            input_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            stats = convert_cmeie_to_kg(input_path, output_path)
            kg = load_medical_kg(output_path)

        self.assertEqual(stats.records, 1)
        self.assertEqual(stats.spo_items, 3)
        self.assertEqual(stats.valid_triples, 2)
        self.assertEqual(stats.written_triples, 1)
        self.assertEqual(kg.relation_count(), 1)
        self.assertEqual(kg.triples[0].to_dict(), {
            "head": "高血压",
            "relation": "药物治疗",
            "tail": "降压药",
            "head_type": "疾病",
            "tail_type": "药物",
            "source": "CMeIE_train",
        })


if __name__ == "__main__":
    unittest.main()
