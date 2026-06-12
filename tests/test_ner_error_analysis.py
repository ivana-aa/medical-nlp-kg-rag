import unittest

from src.ner.error_analysis import analyze_ner_sequences, extract_bio_entities


class BioEntityExtractionTests(unittest.TestCase):
    def test_extract_bio_entities_returns_text_offsets_and_label(self):
        tokens = list("肺炎X咳嗽")
        labels = ["B-DISEASE", "I-DISEASE", "O", "B-SYMPTOM", "I-SYMPTOM"]

        entities = extract_bio_entities(tokens, labels)

        self.assertEqual([entity.to_dict() for entity in entities], [
            {
                "text": "肺炎",
                "label": "DISEASE",
                "start": 0,
                "end": 2,
            },
            {
                "text": "咳嗽",
                "label": "SYMPTOM",
                "start": 3,
                "end": 5,
            },
        ])

    def test_invalid_i_label_starts_a_new_entity(self):
        tokens = list("咳嗽")
        labels = ["I-SYMPTOM", "I-SYMPTOM"]

        entities = extract_bio_entities(tokens, labels)

        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].to_dict(), {
            "text": "咳嗽",
            "label": "SYMPTOM",
            "start": 0,
            "end": 2,
        })


class NERErrorAnalysisTests(unittest.TestCase):
    def test_analyze_ner_sequences_counts_per_label_and_top_errors(self):
        tokens = [list("肺炎X咳嗽X血常规")]
        true_labels = [[
            "B-DISEASE",
            "I-DISEASE",
            "O",
            "B-SYMPTOM",
            "I-SYMPTOM",
            "O",
            "B-EXAM",
            "I-EXAM",
            "I-EXAM",
        ]]
        pred_labels = [[
            "B-DISEASE",
            "I-DISEASE",
            "O",
            "O",
            "O",
            "O",
            "B-TEST",
            "I-TEST",
            "I-TEST",
        ]]

        report = analyze_ner_sequences(tokens, true_labels, pred_labels, max_examples=5)

        self.assertEqual(report["totals"], {"tp": 1, "fp": 1, "fn": 2})
        self.assertEqual(report["per_label"]["DISEASE"]["tp"], 1)
        self.assertEqual(report["per_label"]["SYMPTOM"]["fn"], 1)
        self.assertEqual(report["per_label"]["TEST"]["fp"], 1)
        self.assertEqual(report["top_false_positive_entities"][0]["text"], "血常规")
        self.assertEqual(report["top_false_negative_entities"][0]["text"], "咳嗽")

    def test_analyze_ner_sequences_finds_boundary_and_label_confusion_examples(self):
        tokens = [list("呼吸中枢受累X血常规")]
        true_labels = [[
            "B-SYMPTOM",
            "I-SYMPTOM",
            "I-SYMPTOM",
            "I-SYMPTOM",
            "I-SYMPTOM",
            "I-SYMPTOM",
            "O",
            "B-EXAM",
            "I-EXAM",
            "I-EXAM",
        ]]
        pred_labels = [[
            "B-SYMPTOM",
            "I-SYMPTOM",
            "I-SYMPTOM",
            "I-SYMPTOM",
            "O",
            "O",
            "O",
            "B-TEST",
            "I-TEST",
            "I-TEST",
        ]]

        report = analyze_ner_sequences(tokens, true_labels, pred_labels, max_examples=5)

        self.assertEqual(len(report["boundary_errors"]), 1)
        self.assertEqual(report["boundary_errors"][0]["gold"]["text"], "呼吸中枢受累")
        self.assertEqual(report["boundary_errors"][0]["predicted"]["text"], "呼吸中枢")
        self.assertEqual(len(report["label_confusions"]), 1)
        self.assertEqual(report["label_confusions"][0]["gold_label"], "EXAM")
        self.assertEqual(report["label_confusions"][0]["predicted_label"], "TEST")


if __name__ == "__main__":
    unittest.main()
