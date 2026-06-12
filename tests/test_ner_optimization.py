import unittest

import torch

from src.ner.decoding import apply_entity_confidence_threshold
from src.ner.losses import build_label_loss_weights


class EntityThresholdTests(unittest.TestCase):
    def test_threshold_converts_low_confidence_entity_tokens_to_o(self):
        predictions = torch.tensor([[0, 1, 2, 3]])
        confidences = torch.tensor([[0.99, 0.49, 0.80, 0.30]])
        id2label = {
            0: "O",
            1: "B-DISEASE",
            2: "I-DISEASE",
            3: "B-SYMPTOM",
        }

        filtered = apply_entity_confidence_threshold(
            predictions,
            confidences,
            id2label,
            threshold=0.5,
        )

        self.assertEqual(filtered.tolist(), [[0, 0, 2, 0]])

    def test_zero_threshold_preserves_predictions(self):
        predictions = torch.tensor([[0, 1, 2]])
        confidences = torch.tensor([[0.20, 0.10, 0.05]])
        id2label = {0: "O", 1: "B-DISEASE", 2: "I-DISEASE"}

        filtered = apply_entity_confidence_threshold(
            predictions,
            confidences,
            id2label,
            threshold=0.0,
        )

        self.assertEqual(filtered.tolist(), predictions.tolist())


class WeightedLossTests(unittest.TestCase):
    def test_entity_strategy_weights_only_entity_labels(self):
        label2id = {"O": 0, "B-DISEASE": 1, "I-DISEASE": 2}
        train_labels = [["O", "B-DISEASE", "I-DISEASE", "O"]]

        weights = build_label_loss_weights(
            "entity",
            label2id,
            train_labels,
            entity_weight=0.75,
        )

        self.assertEqual(weights.tolist(), [1.0, 0.75, 0.75])

    def test_inverse_sqrt_gives_rare_labels_higher_weight_with_clamps(self):
        label2id = {"O": 0, "B-DISEASE": 1, "B-EXAM": 2}
        train_labels = [
            ["O", "O", "O", "O", "B-DISEASE"],
            ["O", "O", "O", "O", "B-EXAM"],
        ]

        weights = build_label_loss_weights(
            "inverse_sqrt",
            label2id,
            train_labels,
            min_weight=0.5,
            max_weight=2.0,
        )

        self.assertGreater(float(weights[label2id["B-DISEASE"]]), float(weights[label2id["O"]]))
        self.assertGreaterEqual(float(weights.min()), 0.5)
        self.assertLessEqual(float(weights.max()), 2.0)


if __name__ == "__main__":
    unittest.main()
