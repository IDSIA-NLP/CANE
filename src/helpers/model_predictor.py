from predictor import batch_predict
from loguru import logger

# Connection errors that should trigger retry
CONNECTION_ERRORS = (
    ConnectionError,
    TimeoutError,
    Exception  # Catch-all for API errors
)


def predict_with_llm(llm, input_format, output_format, entities, text, data_examples, retry_attempts=3, task='ner'):
    """Predict a single text, retrying on connection errors. Format errors are not retried."""
    for attempt in range(retry_attempts):
        try:
            prediction = batch_predict(
                llm, input_format, output_format,
                entities, [text], data_examples, task
            )[0]  # Get first item from batch
            
            # Check for fatal errors
            if prediction['fatal_error']:
                logger.warning(f"Fatal error: {prediction['fatal_error']}")
                return {
                    'prediction': prediction['prediction'],
                    'raw_output': prediction['raw_output'],
                    'parsed_output': prediction['parsed_output'],
                    'parse_success': False,
                    'parse_error_type': prediction['fatal_error'],
                    'error_counts': prediction['error_counts'],
                    'prompt': prediction['prompt']
                }
            
            return {
                'prediction': prediction['prediction'],
                'raw_output': prediction['raw_output'],
                'parsed_output': prediction['parsed_output'],
                'parse_success': True,
                'parse_error_type': None,
                'error_counts': prediction['error_counts'],
                'prompt': prediction['prompt']
            }
            
        except CONNECTION_ERRORS as e:
            logger.error(f"Connection error (attempt {attempt + 1}/{retry_attempts}): {e}")
            if attempt == retry_attempts - 1:
                # Last attempt failed - return fallback
                logger.error("All retry attempts exhausted")
                if task == 'ner':
                    fallback = []
                    error_counts = {
                        'malformed_output': 0,
                        'length_mismatch': 0,
                        'invalid_label': 0,
                        'invalid_span': 0,
                        'matching_failed': 0
                    }

                return {
                    'prediction': fallback,
                    'raw_output': '',
                    'parsed_output': None,
                    'parse_success': False,
                    'parse_error_type': 'connection_error',
                    'error_counts': error_counts,
                    'prompt': ''
                }
    
    # Fallback
    if task == 'ner':
        fallback = []
        error_counts = {
            'malformed_output': 0,
            'length_mismatch': 0,
            'invalid_label': 0,
            'invalid_span': 0,
            'matching_failed': 0
        }

    return {
        'prediction': fallback,
        'raw_output': '',
        'parsed_output': None,
        'parse_success': False,
        'parse_error_type': 'unknown_error',
        'error_counts': error_counts,
        'prompt': ''
    }


def predict_batch_with_llm(llm, input_format, output_format, entities, inputs_batch, data_examples, task='ner'):
    """Predict a batch of texts. Returns one result dict per input, with fallbacks if the batch fails."""
    try:
        batch_results = batch_predict(
            llm, input_format, output_format,
            entities, inputs_batch, data_examples, task
        )
        
        # Convert to expected format
        results = []
        for result in batch_results:
            parse_success = (result['fatal_error'] is None)
            parse_error_type = result['fatal_error']
            
            results.append({
                'prediction': result['prediction'],
                'raw_output': result['raw_output'],
                'parsed_output': result['parsed_output'],
                'parse_success': parse_success,
                'parse_error_type': parse_error_type,
                'error_counts': result['error_counts'],
                'prompt': result['prompt']
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        # Return fallbacks for entire batch
        results = []
        for i, text_input in enumerate(inputs_batch):
            if task == 'ner':
                fallback = []
                error_counts = {
                    'malformed_output': 1,
                    'length_mismatch': 0,
                    'invalid_label': 0,
                    'invalid_span': 0,
                    'matching_failed': 0
                }

            results.append({
                'prediction': fallback,
                'raw_output': '',
                'parsed_output': None,
                'parse_success': False,
                'parse_error_type': 'batch_exception',
                'error_counts': error_counts,
                'prompt': ''
            })
        return results