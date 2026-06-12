import unittest

from src.ner.lexicon_postprocess import (
    apply_lexicon_postprocess,
    build_entity_lexicon,
    entities_to_bio_labels,
)


class NERLexiconPostprocessTests(unittest.TestCase):
    def test_build_entity_lexicon_uses_majority_label_and_counts(self):
        tokens = [list("心力衰竭X心力衰竭"), list("心力衰竭")]
        labels = [
            [
                "B-DISEASE",
                "I-DISEASE",
                "I-DISEASE",
                "I-DISEASE",
                "O",
                "B-SYMPTOM",
                "I-SYMPTOM",
                "I-SYMPTOM",
                "I-SYMPTOM",
            ],
            ["B-DISEASE", "I-DISEASE", "I-DISEASE", "I-DISEASE"],
        ]

        lexicon = build_entity_lexicon(tokens, labels)

        self.assertEqual(lexicon["心力衰竭"].label, "DISEASE")
        self.assertEqual(lexicon["心力衰竭"].count, 3)
        self.assertEqual(lexicon["心力衰竭"].label_counts["DISEASE"], 2)

    def test_apply_lexicon_postprocess_corrects_entity_label(self):
        chars = list("患者出现心力衰竭")
        predicted = [
            {"text": "心力衰竭", "type": "SYMPTOM", "start": 4, "end": 8},
        ]
        lexicon = build_entity_lexicon(
            [list("心力衰竭X心力衰竭")],
            [[
                "B-DISEASE",
                "I-DISEASE",
                "I-DISEASE",
                "I-DISEASE",
                "O",
                "B-DISEASE",
                "I-DISEASE",
                "I-DISEASE",
                "I-DISEASE",
            ]],
        )

        result = apply_lexicon_postprocess(chars, predicted, lexicon, min_count=2)

        self.assertEqual(result[0]["type"], "DISEASE")

    def test_apply_lexicon_postprocess_adds_missing_high_frequency_entity(self):
        chars = list("患者疼痛明显")
        lexicon = build_entity_lexicon(
            [list("疼痛疼痛")],
            [["B-SYMPTOM", "I-SYMPTOM", "B-SYMPTOM", "I-SYMPTOM"]],
        )

        result = apply_lexicon_postprocess(
            chars,
            [],
            lexicon,
            min_count=2,
            add_missing=True,
        )

        self.assertEqual(result, [
            {"text": "疼痛", "type": "SYMPTOM", "start": 2, "end": 4}
        ])

    def test_apply_lexicon_postprocess_does_not_add_overlapping_entity(self):
        chars = list("患者胸部疼痛")
        predicted = [
            {"text": "胸部疼痛", "type": "SYMPTOM", "start": 2, "end": 6},
        ]
        lexicon = build_entity_lexicon(
            [list("疼痛疼痛")],
            [["B-SYMPTOM", "I-SYMPTOM", "B-SYMPTOM", "I-SYMPTOM"]],
        )

        result = apply_lexicon_postprocess(
            chars,
            predicted,
            lexicon,
            min_count=2,
            add_missing=True,
        )

        self.assertEqual(result, predicted)

    def test_entities_to_bio_labels_round_trips_postprocessed_entities(self):
        chars = list("患者疼痛")
        entities = [{"text": "疼痛", "type": "SYMPTOM", "start": 2, "end": 4}]

        labels = entities_to_bio_labels(chars, entities)

        self.assertEqual(labels, ["O", "O", "B-SYMPTOM", "I-SYMPTOM"])


if __name__ == "__main__":
    unittest.main()
