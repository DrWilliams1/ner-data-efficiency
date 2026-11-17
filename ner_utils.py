"""Word-aligned NER evaluation and portable result verification.

Portfolio adaptation of the Chapter 11 NER workflow in Hands-On Large
Language Models (Jay Alammar and Maarten Grootendorst, 2024), Apache-2.0.
This file is also embedded in the standalone Colab notebook.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.scheme import IOB2


def write_json(path, value):
    """Write readable JSON and reject undefined/infinite numeric results."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def strict_spans(tags):
    """Return (type, first_word, exclusive_end) entities under strict BIO2.

    B starts an entity. I extends only an existing entity of the same type.
    An orphan I does not create an entity. This preserves invalid transitions
    as errors instead of silently repairing the model's predictions.
    """
    spans = set()
    current_type, start = None, None
    for position, tag in enumerate(list(tags) + ["O"]):
        if tag == "O":
            prefix, entity_type = "O", None
        else:
            prefix, entity_type = tag.split("-", 1)
            if prefix not in {"B", "I"} or not entity_type:
                raise ValueError(f"Expected a BIO2 tag; received {tag!r}")
        # A matching I continues the open span; every other tag closes it.
        if current_type is not None and not (prefix == "I" and entity_type == current_type):
            spans.add((current_type, start, position))
            current_type, start = None, None
        if prefix == "B":
            current_type, start = entity_type, position
    return spans


def score_sequences(gold, predicted, entity_types):
    """Compute exact-span micro scores, per-type counts and word accuracy."""
    if len(gold) != len(predicted) or any(len(a) != len(b) for a, b in zip(gold, predicted)):
        raise ValueError("Gold and predicted sequences must align word for word.")
    if not gold or sum(map(len, gold)) == 0:
        raise ValueError("Evaluation needs at least one scored word.")
    counts = {t: {"tp": 0, "fp": 0, "fn": 0} for t in entity_types}
    correct_words = 0
    invalid_transitions = 0
    for reference, hypothesis in zip(gold, predicted):
        expected, actual = strict_spans(reference), strict_spans(hypothesis)
        for entity_type, _, _ in expected | actual:
            if entity_type not in counts:
                raise ValueError(f"Unknown entity type: {entity_type}")
        for t, _, _ in expected & actual:
            counts[t]["tp"] += 1
        for t, _, _ in actual - expected:
            counts[t]["fp"] += 1
        for t, _, _ in expected - actual:
            counts[t]["fn"] += 1
        correct_words += sum(a == b for a, b in zip(reference, hypothesis))
        previous = "O"
        for tag in hypothesis:
            if tag.startswith("I-") and previous not in {"B-" + tag[2:], tag}:
                invalid_transitions += 1
            previous = tag
    def summarize(c):
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        return {**c, "gold_entities": tp + fn, "predicted_entities": tp + fp,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
    total = {k: sum(v[k] for v in counts.values()) for k in ("tp", "fp", "fn")}
    result = summarize(total)
    # Cross-check our transparent count calculations against the standard library.
    for name, function in (("precision", precision_score), ("recall", recall_score), ("f1", f1_score)):
        reference = function(gold, predicted, mode="strict", scheme=IOB2, zero_division=0)
        if not np.isclose(result[name], reference, rtol=0, atol=1e-12):
            raise AssertionError(f"Span count disagrees with seqeval for {name}.")
    result.update({"n_sentences": len(gold), "n_words": sum(map(len, gold)),
                   "word_accuracy": correct_words / sum(map(len, gold)),
                   "invalid_predicted_i_transitions": invalid_transitions,
                   "per_type": {t: summarize(c) for t, c in counts.items()},
                   "scheme": "strict IOB2; exact word span and entity type"})
    return result


def align_first_subword(batch, tokenizer, max_length=512):
    """Label each original word once; ignore continuation and special tokens."""
    # Avoid silent truncation, which can turn a partial entity into a full one.
    encoded = tokenizer(batch["tokens"], is_split_into_words=True, truncation=False)
    aligned = []
    for row, tags in enumerate(batch["ner_tags"]):
        word_ids = encoded.word_ids(batch_index=row)
        if len(word_ids) > max_length:
            raise ValueError(f"Input has {len(word_ids)} tokens; design a windowing policy before proceeding.")
        if len(tags) != len(batch["tokens"][row]):
            raise ValueError("Dataset word and label counts differ.")
        labels, seen = [], set()
        for word_id in word_ids:
            if word_id is None or word_id in seen:
                labels.append(-100)
            else:
                labels.append(tags[word_id])
                seen.add(word_id)
        if seen != set(range(len(tags))):
            raise ValueError("Tokenization did not preserve every original word.")
        aligned.append(labels)
    encoded["labels"] = aligned
    return encoded


def nested_indices(n_rows, fractions, seed):
    """Use prefixes of one permutation so smaller samples are nested."""
    order = np.random.default_rng(seed).permutation(n_rows)
    return {str(int(f * 100)): order[:max(1, int(np.floor(n_rows * f)))].tolist()
            for f in fractions}


def prediction_arrays(logits, labels, label_names):
    """Remove ignored positions while keeping sentence boundaries and scores."""
    if isinstance(logits, tuple):
        logits = logits[0]
    offsets, gold, probabilities = [0], [], []
    for row_logits, row_labels in zip(logits, labels):
        mask = row_labels != -100
        values = np.asarray(row_logits[mask], dtype=np.float64)
        values -= values.max(axis=1, keepdims=True)
        probability = np.exp(values)
        probability /= probability.sum(axis=1, keepdims=True)
        probabilities.append(probability.astype(np.float32))
        gold.extend(np.asarray(row_labels[mask], dtype=np.int16).tolist())
        offsets.append(len(gold))
    probability = np.concatenate(probabilities, axis=0)
    return {"row_ids": np.arange(len(offsets) - 1, dtype=np.int32),
            "offsets": np.asarray(offsets, dtype=np.int64),
            "gold_ids": np.asarray(gold, dtype=np.int16),
            "predicted_ids": probability.argmax(axis=1).astype(np.int16),
            "probabilities": probability,
            "label_names": np.asarray(label_names)}


def sequences_from_arrays(arrays):
    """Restore a list of complete word-label sequences from a saved NPZ."""
    names = arrays["label_names"].tolist()
    offsets = arrays["offsets"]
    if offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
        raise ValueError("Invalid sentence offsets.")
    gold_ids, predicted_ids = arrays["gold_ids"], arrays["predicted_ids"]
    probabilities = arrays["probabilities"]
    if int(offsets[-1]) != len(gold_ids) or len(gold_ids) != len(predicted_ids):
        raise ValueError("Saved labels do not match the sentence offsets.")
    if len(arrays["row_ids"]) != len(offsets) - 1:
        raise ValueError("Saved row identifiers do not match the sentence count.")
    if probabilities.shape != (len(gold_ids), len(names)):
        raise ValueError("Saved probability dimensions are inconsistent.")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("Invalid saved probabilities.")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Saved probabilities do not sum to one.")
    if not np.array_equal(probabilities.argmax(axis=1), predicted_ids):
        raise ValueError("Saved predictions differ from the probability argmax.")
    if np.any(gold_ids < 0) or np.any(gold_ids >= len(names)):
        raise ValueError("Saved gold label outside the label vocabulary.")
    def unpack(ids):
        return [[names[int(x)] for x in ids[start:end]] for start, end in zip(offsets[:-1], offsets[1:])]
    return unpack(gold_ids), unpack(predicted_ids)


def metrics_from_arrays(arrays):
    gold, predicted = sequences_from_arrays(arrays)
    types = sorted({tag[2:] for tag in arrays["label_names"].tolist() if tag != "O"})
    return score_sequences(gold, predicted, types)


def error_rows(arrays):
    """Create a text-free worksheet of unmatched gold and predicted entities.

    An incorrect span may generate both a missed-entity and an extra-entity
    row. These are metric contributions, not mutually exclusive root causes.
    """
    gold, predicted = sequences_from_arrays(arrays)
    rows = []
    for i, (reference, hypothesis) in enumerate(zip(gold, predicted)):
        expected, actual = strict_spans(reference), strict_spans(hypothesis)
        for kind, spans in (("missed_gold_entity", expected - actual), ("extra_predicted_entity", actual - expected)):
            for entity_type, start, end in sorted(spans):
                offset = int(arrays["offsets"][i])
                probability = arrays["probabilities"][offset + start:offset + end]
                rows.append({"validation_row_id": int(arrays["row_ids"][i]),
                             "error_kind": kind, "entity_type": entity_type,
                             "start_word": start, "end_word_exclusive": end,
                             "mean_predicted_token_score": float(probability.max(axis=1).mean()),
                             "my_observation": ""})
    return sorted(rows, key=lambda r: (-r["mean_predicted_token_score"], r["validation_row_id"], r["start_word"], r["error_kind"]))


def choose_fraction(records):
    """Select by mean validation F1; break exact ties with less training data."""
    fractions = sorted({r["fraction_percent"] for r in records})
    means = {f: float(np.mean([r["metrics"]["f1"] for r in records if r["fraction_percent"] == f])) for f in fractions}
    chosen = min(fractions, key=lambda f: (-means[f], f))
    return chosen, means


def normalized_sentence_hash(tokens):
    """Hash normalized words for an exact-overlap audit without exporting text."""
    value = " ".join(" ".join(tokens).casefold().split())
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_manifest(run_dir):
    """Record file sizes and hashes; integrity alone does not prove correctness."""
    run_dir = Path(run_dir)
    manifest = {str(p.relative_to(run_dir)): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                            "size_bytes": p.stat().st_size}
                for p in sorted(run_dir.rglob("*")) if p.is_file() and p.name != "MANIFEST_SHA256.json"}
    write_json(run_dir / "MANIFEST_SHA256.json", manifest)
    return manifest
