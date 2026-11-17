"""Regression cases for entity boundaries, alignment, and selection."""
import unittest
import numpy as np
from ner_utils import (align_first_subword, strict_spans, score_sequences,
                       nested_indices, prediction_arrays, metrics_from_arrays,
                       sequences_from_arrays, choose_fraction, error_rows)

TYPES = ["LOC", "ORG", "PER"]
NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


class Encoded(dict):
    def word_ids(self, batch_index):
        return [None, 0, 0, 1, None]


class NerRegressionTests(unittest.TestCase):
    def test_partial_name_does_not_count_as_correct_entity(self):
        score = score_sequences([["B-PER", "I-PER"]], [["B-PER", "O"]], TYPES)
        self.assertEqual((score["tp"], score["fp"], score["fn"]), (0, 1, 1))
        self.assertEqual(score["f1"], 0)
        self.assertEqual(score["word_accuracy"], 0.5)

    def test_wrong_type_same_span_is_wrong(self):
        score = score_sequences([["B-PER", "I-PER"]], [["B-ORG", "I-ORG"]], TYPES)
        self.assertEqual((score["tp"], score["fp"], score["fn"]), (0, 1, 1))

    def test_entities_do_not_cross_sentence_boundaries(self):
        score = score_sequences([["B-PER"], ["B-PER"]], [["B-PER"], ["I-PER"]], TYPES)
        self.assertEqual((score["tp"], score["fp"], score["fn"]), (1, 0, 1))
        self.assertEqual(score["invalid_predicted_i_transitions"], 1)

    def test_invalid_i_tag_is_not_repaired(self):
        self.assertEqual(strict_spans(["I-PER", "I-PER", "O"]), set())
        score_sequences([["B-PER", "I-PER", "O"]], [["I-PER", "I-PER", "O"]], TYPES)

    def test_exact_entity_and_adjacent_same_type(self):
        tags = ["B-PER", "I-PER", "B-PER", "O", "B-ORG"]
        self.assertEqual(strict_spans(tags), {("PER", 0, 2), ("PER", 2, 3), ("ORG", 4, 5)})
        self.assertEqual(score_sequences([tags], [tags], TYPES)["f1"], 1)

    def test_no_entities_has_zero_f1_and_explicit_zero_support(self):
        s = score_sequences([["O"]], [["O"]], TYPES)
        self.assertEqual((s["f1"], s["gold_entities"], s["predicted_entities"]), (0, 0, 0))

    def test_subwords_and_special_tokens_are_ignored(self):
        fake_tokenizer = lambda *a, **k: Encoded(input_ids=[[101, 1, 2, 3, 102]])
        result = align_first_subword({"tokens": [["Maarten", "."]], "ner_tags": [[1, 0]]}, fake_tokenizer)
        self.assertEqual(result["labels"], [[-100, 1, -100, 0, -100]])
        with self.assertRaises(ValueError):
            align_first_subword({"tokens": [["Maarten", "."]], "ner_tags": [[1, 0]]}, fake_tokenizer, 4)

    def test_nested_subsets_are_reproducible_and_unique(self):
        a = nested_indices(14041, [0.1, 0.25, 1.0], 42)
        self.assertEqual(a, nested_indices(14041, [0.1, 0.25, 1.0], 42))
        self.assertEqual([len(a[k]) for k in ["10", "25", "100"]], [1404, 3510, 14041])
        self.assertEqual(a["10"], a["25"][:1404])
        self.assertEqual(len(set(a["100"])), 14041)

    def test_saved_probabilities_reconstruct_sequences_and_detect_tampering(self):
        labels = np.array([[-100, 1, 2, -100], [-100, 5, 0, -100]])
        logits = np.zeros((2, 4, len(NAMES)))
        logits[0, 1, 1], logits[0, 2, 0], logits[1, 1, 5] = 8, 8, 8
        arrays = prediction_arrays(logits, labels, NAMES)
        self.assertEqual(metrics_from_arrays(arrays)["tp"], 1)
        self.assertEqual(len(error_rows(arrays)), 2)
        arrays["predicted_ids"][0] = 0
        with self.assertRaises(ValueError): sequences_from_arrays(arrays)

    def test_selection_uses_mean_validation_f1_and_smaller_fraction_tie_break(self):
        records = [{"fraction_percent": f, "metrics": {"f1": x}} for f, x in [(10, .8), (10, .8), (100, .9), (100, .6)]]
        self.assertEqual(choose_fraction(records)[0], 10)
        self.assertEqual(choose_fraction([{"fraction_percent": f, "metrics": {"f1": .8}} for f in [100, 10]])[0], 10)

    def test_random_sequences_agree_with_seqeval_strict(self):
        rng = np.random.default_rng(101)
        for _ in range(200):
            gold = [rng.choice(NAMES, size=12).tolist() for _ in range(3)]
            pred = [rng.choice(NAMES, size=12).tolist() for _ in range(3)]
            score_sequences(gold, pred, TYPES)


if __name__ == "__main__":
    unittest.main()
