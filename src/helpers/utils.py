import pandas as pd
import json
from loguru import logger


def save_results_and_config(config, output_results_name, results, ckpt=False):
    """Save results to a TSV file and config to a JSON file."""
    df = pd.DataFrame(results)
    if ckpt:
        output_results_name += "_ckpt"
    df.to_csv(f"{config['results_path']}{output_results_name}.tsv", index=False, sep="\t")
    logger.info(f"Saved results to: {config['results_path']}{output_results_name}.tsv")
    with open(f"{config['results_path']}{output_results_name}_config.json", "w") as outfile:
        json.dump(config, outfile, indent=4)


def save_experiment_metadata(config, output_results_name, examples, entities, input_format, output_format):
    """Save experiment metadata including examples used (once per experiment)."""
    metadata = {
        'task': config['task'],
        'dataset': output_results_name.split('_')[2] if '_' in output_results_name else 'unknown',
        'model': output_results_name.split('_')[3] if '_' in output_results_name else 'unknown',
        'input_format': input_format,
        'output_format': output_format,
        'num_examples': config['num_examples'],
        'num_test_samples': config['num_test_samples'],
        'batch_size': config.get('batch_size', 1),
        'examples_used': examples,
        'entities': entities
    }
    
    output_path = f"{config['results_path']}{output_results_name}_metadata.json"
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved experiment metadata to: {output_path}")


def save_detailed_results(config, output_results_name, detailed_predictions):
    """Save detailed per-prediction results to JSONL file."""
    output_path = f"{config['results_path']}{output_results_name}_detailed.jsonl"
    with open(output_path, 'w') as f:
        for pred in detailed_predictions:
            f.write(json.dumps(pred) + '\n')
    logger.info(f"Saved detailed results to: {output_path}")


def save_failure_analysis(config, output_results_name, detailed_predictions):
    """Save failure analysis grouping errors by type."""
    # Aggregate error counts across all predictions
    total_error_counts = {}
    failures_by_type = {}
    
    for pred in detailed_predictions:
        error_counts = pred.get('error_counts', {})
        
        for error_type, count in error_counts.items():
            # Aggregate totals
            if error_type not in total_error_counts:
                total_error_counts[error_type] = 0
            total_error_counts[error_type] += count
            
            # Group failures by type (only if count > 0)
            if count > 0:
                if error_type not in failures_by_type:
                    failures_by_type[error_type] = []
                
                failures_by_type[error_type].append({
                    'text': pred.get('text', ''),
                    'ground_truth': pred['ground_truth'],
                    'prediction': pred['prediction'],
                    'raw_output': pred['raw_output'],
                    'parsed_output': pred['parsed_output'],
                    'prompt': pred.get('prompt', ''),
                    'error_count': count
                })
    
    failure_analysis = {
        'total_predictions': len(detailed_predictions),
        'successful_parses': sum(1 for p in detailed_predictions if p['parse_success']),
        'failed_parses': sum(1 for p in detailed_predictions if not p['parse_success']),
        'parse_success_rate': sum(1 for p in detailed_predictions if p['parse_success']) / len(detailed_predictions) if detailed_predictions else 0,
        'total_error_counts': total_error_counts,
        'failures_by_type': failures_by_type
    }
    
    output_path = f"{config['results_path']}{output_results_name}_failures.json"
    with open(output_path, 'w') as f:
        json.dump(failure_analysis, f, indent=2)
    logger.info(f"Saved failure analysis to: {output_path}")
    
    return failure_analysis


def save_extended_metrics(config, output_results_name, extended_metrics):
    """Save extended evaluation metrics (matching schemes, per-entity breakdown)."""
    output_path = f"{config['results_path']}{output_results_name}_extended_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(extended_metrics, f, indent=2)
    logger.info(f"Saved extended metrics to: {output_path}")


def convert_results(config, macro_avg, weighted_avg, micro_avg, dataset, model_name, input_format, output_format, failure_stats=None, extended_metrics=None):
    """Convert evaluation metrics to a standardized result format."""
    result = {
        "task": config['task'],
        "num_test_samples": config['num_test_samples'],
        "num_examples": config['num_examples'],
        "batch_size": config.get('batch_size', 1),
        "model_name": model_name,
        "input_format": input_format,
        "output_format": output_format,
        "dataset": dataset,
        "macro_avg_precision": macro_avg['precision'],
        "macro_avg_recall": macro_avg['recall'],
        "macro_avg_f1": macro_avg['f1-score'],
        "macro_avg_support": macro_avg['support'],
        "weighted_avg_precision": weighted_avg['precision'],
        "weighted_avg_recall": weighted_avg['recall'],
        "weighted_avg_f1": weighted_avg['f1-score'],
        "weighted_avg_support": weighted_avg['support'],
        "micro_avg_precision": micro_avg['precision'],
        "micro_avg_recall": micro_avg['recall'],
        "micro_avg_f1": micro_avg['f1-score'],
        "micro_avg_support": micro_avg['support']
    }
    
    # Add parse statistics if available
    if failure_stats:
        result["parse_success_rate"] = failure_stats['parse_success_rate']
        result["total_predictions"] = failure_stats['total_predictions']
        result["successful_parses"] = failure_stats['successful_parses']
        result["failed_parses"] = failure_stats['failed_parses']
        
        # Add granular error counts
        total_error_counts = failure_stats.get('total_error_counts', {})
        for error_type, count in total_error_counts.items():
            result[f"error_{error_type}"] = count
    
    # Add extended metrics summary (for NER)
    if extended_metrics and 'strict' in extended_metrics:
        result["strict_f1"] = extended_metrics['strict']['f1']
        result["partial_f1"] = extended_metrics['partial']['f1']
        result["exact_f1"] = extended_metrics['exact']['f1']
        result["ent_type_f1"] = extended_metrics['ent_type']['f1']
        result["strict_correct"] = extended_metrics['strict']['correct']
        result["strict_incorrect"] = extended_metrics['strict']['incorrect']
        result["strict_partial"] = extended_metrics['strict']['partial']
        result["strict_missed"] = extended_metrics['strict']['missed']
        result["strict_spurious"] = extended_metrics['strict']['spurious']
        result["strict_possible"] = extended_metrics['strict']['possible']
        result["strict_actual"] = extended_metrics['strict']['actual']
    
    return result