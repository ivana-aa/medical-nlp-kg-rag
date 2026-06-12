import unittest

from src.ner.lexicon_postprocess import LexiconEntry
from src.ner.predict import MedicalNERPredictor


class MedicalNERPredictorPostprocessTests(unittest.TestCase):
    def test_postprocess_preserves_entities_when_no_lexicon_is_configured(self):
        predictor = MedicalNERPredictor.__new__(MedicalNERPredictor)
        predictor.entity_lexicon = None
        predictor.lexicon_min_count = 5
        predictor.lexicon_add_missing = False
        entities = [{"text": "心力衰竭", "type": "SYMPTOM", "start": 4, "end": 8}]

        result = predictor._postprocess_entities(list("患者出现心力衰竭"), entities)

        self.assertEqual(result, entities)

    def test_postprocess_corrects_labels_with_configured_training_lexicon(self):
        predictor = MedicalNERPredictor.__new__(MedicalNERPredictor)
        predictor.entity_lexicon = {
            "心力衰竭": LexiconEntry(
                text="心力衰竭",
                label="DISEASE",
                count=8,
                label_counts={"DISEASE": 8},
            )
        }
        predictor.lexicon_min_count = 5
        predictor.lexicon_add_missing = False
        entities = [{"text": "心力衰竭", "type": "SYMPTOM", "start": 4, "end": 8}]

        result = predictor._postprocess_entities(list("患者出现心力衰竭"), entities)

        self.assertEqual(result, [
            {"text": "心力衰竭", "type": "DISEASE", "start": 4, "end": 8}
        ])

    def test_postprocess_can_add_missing_entities_when_explicitly_enabled(self):
        predictor = MedicalNERPredictor.__new__(MedicalNERPredictor)
        predictor.entity_lexicon = {
            "疼痛": LexiconEntry(
                text="疼痛",
                label="SYMPTOM",
                count=10,
                label_counts={"SYMPTOM": 10},
            )
        }
        predictor.lexicon_min_count = 5
        predictor.lexicon_add_missing = True

        result = predictor._postprocess_entities(list("患者疼痛明显"), [])

        self.assertEqual(result, [
            {"text": "疼痛", "type": "SYMPTOM", "start": 2, "end": 4}
        ])


if __name__ == "__main__":
    unittest.main()
