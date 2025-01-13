def extract_entities(sequence):
    entities = []
    start = None
    for i, label in enumerate(sequence):
        if label.startswith("B"):
            start = i
        elif label.startswith("E") and start is not None:
            entities.append((start, i))
            start = None
        elif label == "O":
            start = None
    return entities

def is_within_tolerance(true_entity, pred_entity, tolerance):
    true_start, true_end = true_entity
    pred_start, pred_end = pred_entity
    
    return abs(true_start - pred_start) <= tolerance and abs(true_end - pred_end) <= tolerance

def metrics(y_true, y_pred):
    assert len(y_true) == len(y_pred), "dismatched length between y_true and y_pred."

    results = {}

    for tolerance in [0, 1, 2, 3]:
        true_entities = 0
        predicted_entities = 0
        correct_entities = 0

        for true_seq, pred_seq in zip(y_true, y_pred):
            true_entity_set = set(extract_entities(true_seq))
            pred_entity_set = set(extract_entities(pred_seq))

            true_entities += len(true_entity_set)
            predicted_entities += len(pred_entity_set)

            for pred_entity in pred_entity_set:
                if any(is_within_tolerance(true_entity, pred_entity, tolerance) for true_entity in true_entity_set):
                    correct_entities += 1

        recall = correct_entities / true_entities if true_entities > 0 else 0.0
        precision = correct_entities / predicted_entities if predicted_entities > 0 else 0.0
        f1_score = (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0.0

        results[tolerance] = {
            "recall": recall,
            "precision": precision,
            "f1_score": f1_score
        }

    return results

