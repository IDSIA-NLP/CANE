#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script to run the NER evaluation pipeline.
"""

from loaders.config_loader import load_config
import os
from decouple import config as config_decouple

# Set HuggingFace token for private models/datasets
os.environ['HF_TOKEN'] = config_decouple('HF_TOKEN')

# Load config and set cache FIRST
config = load_config("./config.yml")
if 'cache_dir' in config:
    cache = config['cache_dir']
    os.environ['HF_HOME'] = os.path.join(cache, 'huggingface')
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache, 'huggingface')
    os.environ['HF_DATASETS_CACHE'] = os.path.join(cache, 'huggingface/datasets')
    os.environ['VLLM_CACHE_DIR'] = os.path.join(cache, 'vllm')
    os.environ['TORCH_HOME'] = os.path.join(cache, 'torch')
    os.environ['XDG_CACHE_HOME'] = cache  # General Linux cache

from loaders.data_loader.dataset_loader import get_dataset_and_entities
from loaders.model_loader import load_model
from helpers.model_predictor import predict_batch_with_llm
from evaluator import evaluate_ner
from helpers.utils import save_results_and_config, convert_results, save_detailed_results, save_failure_analysis, save_experiment_metadata, save_extended_metrics
from loguru import logger
import datetime
import gc

# Set up logging
logger.add("../logs/run.log", rotation="500 KB")


def main(config):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_results_name = f"results_task-{config['task']}"
    
    # Create timestamped results folder
    results_folder = os.path.join(config['results_path'], f"{timestamp}_results-{config['task']}")
    os.makedirs(results_folder, exist_ok=True)
    
    # Update config to use this folder
    config['results_path'] = results_folder + '/'
    
    results = []
    batch_size = config.get('batch_size', 32)

    # Pre-load all datasets to avoid reloading for each model
    logger.info("Pre-loading all datasets...")
    datasets_cache = {}
    for dataset in config[config['task']]['datasets']:
        logger.info(f"Loading dataset: {dataset}")
        num_examples = config.get('dataset_overrides', {}).get(dataset, {}).get('num_examples', config['num_examples'])
        logger.info(f"Using num_examples={num_examples} for dataset: {dataset}")
        if config['task'] == 'ner':
            data, entities = get_dataset_and_entities(
                dataset, config['task'], config['seed'], config['num_test_samples'], 
                num_examples, config[config['task']]['use_semantic_labels'], 
                config['resource_path'], config['examples_file'], 
                config['labels_file']
            )
            datasets_cache[dataset] = {'data': data, 'entities': entities}

    logger.info(f"Loaded {len(datasets_cache)} datasets")

    # Model-first loop structure for efficient GPU usage
    num_models = len(config['model_names'])
    for model_idx, model_name in enumerate(config['model_names']):
        logger.info(f"=== Loading model {model_idx + 1}/{num_models}: {model_name} ===")
        
        # Load model once for all datasets and formats
        enable_thinking = model_name.endswith(":thinking")
        if enable_thinking:
            model_name_clean = model_name.rsplit(":thinking", 1)[0]
        else:
            model_name_clean = model_name

        llm = load_model(model_name_clean, enable_thinking=enable_thinking)
        logger.info(f"Model {model_name} loaded successfully")
        
        # Run all experiments for this model
        for dataset in config[config['task']]['datasets']:
            logger.info(f"Running dataset: {dataset} with model: {model_name}")
            
            # Get cached dataset
            cached = datasets_cache[dataset]
            data = cached['data']
            entities = cached['entities']

            for input_format in config[config['task']]['input_formats']:
                for output_format in config[config['task']]['output_formats']:
                    labels_and_predictions = {"labels": [], "predictions": []}
                    detailed_predictions = []
                    
                    # Save experiment metadata (examples used - once per experiment)
                    exp_name = f"{output_results_name}_{dataset}_{model_name}_{input_format}_{output_format}"
                    # Sanitize exp name
                    exp_name = exp_name.replace("/", "_").replace(":", "_")
                    save_experiment_metadata(
                        config, exp_name, data["examples"], 
                        entities, input_format, output_format
                    )
                    
                    # Batch processing
                    test_data = data["test"]
                    for batch_start in range(0, len(test_data), batch_size):
                        batch_end = min(batch_start + batch_size, len(test_data))
                        batch = test_data[batch_start:batch_end]
                        
                        logger.debug(f"Processing batch {batch_start//batch_size + 1}: items {batch_start}-{batch_end}")
                        
                        if config['task'] == 'ner':
                            # Prepare batch inputs (text strings)
                            batch_inputs = [item["text"] for item in batch]
                            
                            # Batch predict
                            batch_results = predict_batch_with_llm(
                                llm, input_format, output_format,
                                entities, batch_inputs, data["examples"],
                                config['task']
                            )

                            # Store results with detailed tracking
                            for item, result in zip(batch, batch_results):
                                prediction = result['prediction']

                                # nervaluate-disco format
                                labels_and_predictions["labels"].append({
                                    'text': item["text"],
                                    'entities': item["entities"]
                                })
                                labels_and_predictions["predictions"].append({
                                    'text': item["text"],
                                    'entities': prediction if prediction else []  # Handle None/parse failures
                                })
                                
                                # Check if prediction is correct
                                is_correct = (prediction == item["entities"]) if result['parse_success'] else False
                                
                                # Store detailed prediction info with error_counts
                                detailed_predictions.append({
                                    'text': item["text"],
                                    'ground_truth': item["entities"],
                                    'prediction': prediction if prediction else [],
                                    'raw_output': result['raw_output'],
                                    'parsed_output': result['parsed_output'],
                                    'parse_success': result['parse_success'],
                                    'parse_error_type': result['parse_error_type'],
                                    'error_counts': result['error_counts'],
                                    'is_correct': is_correct,
                                    'prompt': result['prompt']
                                })

                    # Save detailed results and failure analysis
                    save_detailed_results(config, exp_name, detailed_predictions)
                    failure_stats = save_failure_analysis(config, exp_name, detailed_predictions)
                    
                    # Evaluate predictions
                    if config['task'] == 'ner':
                        macro_avg, weighted_avg, micro_avg, extended_metrics = evaluate_ner(labels_and_predictions)
                        # Save extended metrics to separate file
                        save_extended_metrics(config, exp_name, extended_metrics)

                    results.append(convert_results(
                        config, macro_avg, weighted_avg, micro_avg, 
                        dataset, model_name, input_format, output_format, 
                        failure_stats, extended_metrics
                    ))
        
        # Save checkpoint after each model completes
        save_results_and_config(config, output_results_name, results, ckpt=True)
        
        # Unload model and free GPU memory before loading next model
        logger.info(f"Unloading model: {model_name}")
        del llm
        gc.collect()
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
        except ImportError:
            pass
        
    save_results_and_config(config, output_results_name, results)
    logger.info("Done!")


if __name__ == '__main__':
    logger.info('Start ...')
    config = load_config("./config.yml")
    main(config)