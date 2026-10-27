from nervaluate import Evaluator
from loguru import logger

def evaluate_ner(labels_and_predictions):
    """
    Evaluate NER predictions with nervaluate-disco.

    labels_and_predictions has 'labels' and 'predictions', each a list of
    {'text', 'entities'} dicts.
    """
    true_entities = labels_and_predictions["labels"]
    pred_entities = labels_and_predictions["predictions"]
    
    # Get unique labels
    all_labels = set()
    for doc in true_entities:
        for entity in doc.get('entities', []):
            all_labels.add(entity['label'])
    
    tags = list(all_labels) if all_labels else ['DUMMY']
    
    if not all_labels:
        logger.warning("No entity labels found")
        empty = _empty_metrics()
        return empty, empty, empty, {}
    
    # Use nervaluate-disco with span_dict loader
    evaluator = Evaluator(true_entities, pred_entities, tags=tags, loader='span_dict')
    
    results = evaluator.evaluate()

    # Values are EvaluationResult dataclasses
    overall_results = results['overall']
    per_entity_results = results['entities']

    strict_results = overall_results['strict']
    partial_results = overall_results['partial']
    exact_results = overall_results['exact']
    ent_type_results = overall_results['ent_type']
    
    # Calculate per-entity-type metrics for macro/weighted averaging
    entity_metrics = []
    for tag in tags:
        if tag in per_entity_results and 'strict' in per_entity_results[tag]:
            tag_results = per_entity_results[tag]['strict']
            entity_metrics.append({
                'label': tag,
                'precision': tag_results.precision,
                'recall': tag_results.recall,
                'f1': tag_results.f1,
                'support': tag_results.actual,
                'correct': tag_results.correct,
                'incorrect': tag_results.incorrect,
                'partial': tag_results.partial,
                'missed': tag_results.missed,
                'spurious': tag_results.spurious
            })
    
    # Calculate macro average (unweighted average across entity types)
    if entity_metrics:
        macro_avg = {
            'precision': sum(m['precision'] for m in entity_metrics) / len(entity_metrics),
            'recall': sum(m['recall'] for m in entity_metrics) / len(entity_metrics),
            'f1-score': sum(m['f1'] for m in entity_metrics) / len(entity_metrics),
            'support': sum(m['support'] for m in entity_metrics)
        }
    else:
        macro_avg = _empty_metrics()
    
    # Calculate weighted average (weighted by support per entity type)
    total_support = sum(m['support'] for m in entity_metrics)
    if total_support > 0:
        weighted_avg = {
            'precision': sum(m['precision'] * m['support'] for m in entity_metrics) / total_support,
            'recall': sum(m['recall'] * m['support'] for m in entity_metrics) / total_support,
            'f1-score': sum(m['f1'] * m['support'] for m in entity_metrics) / total_support,
            'support': total_support
        }
    else:
        weighted_avg = _empty_metrics()
    
    # Micro average (global metrics from strict evaluation)
    micro_avg = {
        'precision': strict_results.precision,
        'recall': strict_results.recall,
        'f1-score': strict_results.f1,
        'support': strict_results.actual
    }
    
    # Extended metrics as plain dicts for serialization
    def _result_to_dict(result):
        return {
            'precision': result.precision,
            'recall': result.recall,
            'f1': result.f1,
            'correct': result.correct,
            'incorrect': result.incorrect,
            'partial': result.partial,
            'missed': result.missed,
            'spurious': result.spurious,
            'possible': result.possible,
            'actual': result.actual
        }
    
    extended_metrics = {
        'strict': _result_to_dict(strict_results),
        'partial': _result_to_dict(partial_results),
        'exact': _result_to_dict(exact_results),
        'ent_type': _result_to_dict(ent_type_results),
        'per_entity': entity_metrics
    }
    
    # Log comprehensive results
    logger.info("=== NER Evaluation Results ===")
    logger.info(f"Macro avg: P={macro_avg['precision']:.3f}, R={macro_avg['recall']:.3f}, F1={macro_avg['f1-score']:.3f}")
    logger.info(f"Weighted avg: P={weighted_avg['precision']:.3f}, R={weighted_avg['recall']:.3f}, F1={weighted_avg['f1-score']:.3f}")
    logger.info(f"Micro avg: P={micro_avg['precision']:.3f}, R={micro_avg['recall']:.3f}, F1={micro_avg['f1-score']:.3f}")
    
    # Log matching scheme comparison
    logger.info("\n=== Matching Scheme Comparison ===")
    logger.info(f"Strict (exact span + type): F1={strict_results.f1:.3f}")
    logger.info(f"Partial (overlapping spans): F1={partial_results.f1:.3f}")
    logger.info(f"Exact (exact span, ignore type): F1={exact_results.f1:.3f}")
    logger.info(f"Type (correct type, ignore span): F1={ent_type_results.f1:.3f}")
    
    # Log error breakdown
    logger.info("\n=== Error Analysis (Strict) ===")
    logger.info(f"Correct: {strict_results.correct}")
    logger.info(f"Incorrect: {strict_results.incorrect}")
    logger.info(f"Partial: {strict_results.partial}")
    logger.info(f"Missed: {strict_results.missed}")
    logger.info(f"Spurious: {strict_results.spurious}")
    
    # Log per-entity metrics
    logger.info("\n=== Per-Entity-Type Metrics ===")
    for m in entity_metrics:
        logger.info(
            f"{m['label']:15s} | "
            f"P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1']:.3f} | "
            f"support={m['support']:3d} | "
            f"correct={m['correct']:3d}, incorrect={m['incorrect']:3d}, "
            f"partial={m['partial']:3d}, missed={m['missed']:3d}, spurious={m['spurious']:3d}"
        )
    
    return macro_avg, weighted_avg, micro_avg, extended_metrics


def _empty_metrics():
    """Return empty metrics dict for edge cases."""
    return {
        'precision': 0.0,
        'recall': 0.0,
        'f1-score': 0.0,
        'support': 0
    }