from loguru import logger
from helpers.context_builder import build_ner_context
from loaders.model_loader import load_model
from formatters.formatter_factory import FormatterFactory
from parsers.parser_factory import ParserFactory
from templates.prompt_templates import ner_prompt
from helpers.validation import validate_ner_output


def format_input(task, input_formatter, input_text):
    """Format the input text with the given formatter."""
    return input_formatter.format(input_text)


def process_output(task, result, entities):
    """Keep only well-formed entities with a valid label."""
    if task == "ner":
        # result should be a list of entity dicts with 'spans', 'label', 'text'
        if not isinstance(result, list):
            return []
        
        # Validate each entity has required fields
        validated = []
        for entity in result:
            if isinstance(entity, dict) and 'spans' in entity and 'label' in entity:
                # Check if label is valid
                if entity['label'] in entities or any(entity['label'] == e.split('-')[1] if '-' in e else e for e in entities):
                    validated.append(entity)
        return validated
    
    return result


def predict(llm, input_format, output_format, entities, input_text, data_examples, task="ner"):
    """Predict entities for a single text."""
    # Get formatters
    input_formatter = FormatterFactory.get_formatter(input_format, task, 'input')
    output_formatter = FormatterFactory.get_formatter(output_format, task, 'output')
    
    # Get parser
    parser = ParserFactory.get_parser(task, output_format)
    
    # Build context and format input
    if task == "ner":
        context = build_ner_context(input_format, output_format, entities, data_examples)
        formatted_input = format_input(task, input_formatter, input_text)
        
        # Format prompt
        prompt_text = ner_prompt.format(input=formatted_input, context=context)
        logger.debug(f"Prompt: {prompt_text}")

    # Generate
    raw_output = llm.generate(prompt_text)
    logger.debug(f"Raw output: {raw_output}")
    
    # Parse
    result = parser.parse(raw_output)
    logger.debug(f"Parsed result: {result}")
    
    return process_output(task, result, entities)


def batch_predict(llm, input_format, output_format, entities, inputs_batch, data_examples, task="ner"):
    """Predict a batch of texts, returning prediction, raw output, prompt and validation info per item."""
    # Get formatters
    input_formatter = FormatterFactory.get_formatter(input_format, task, 'input')
    output_formatter = FormatterFactory.get_formatter(output_format, task, 'output')
    
    # Get parser
    parser = ParserFactory.get_parser(task, output_format)
    
    # Build context (same for all items in batch)
    if task == "ner":
        context = build_ner_context(input_format, output_format, entities, data_examples)

    # Format all prompts
    prompt_texts = []
    for input_item in inputs_batch:
        if task == "ner":
            formatted_input = format_input(task, input_formatter, input_item)
            prompt_text = ner_prompt.format(input=formatted_input, context=context)

        prompt_texts.append(prompt_text)
    
    # Batch generate
    raw_outputs = llm.generate_batch(prompt_texts)
    logger.debug(f"Batch raw outputs: {raw_outputs}")
    
    # Parse all outputs, validate, and return results
    results = []
    for input_item, prompt_text, raw_output in zip(inputs_batch, prompt_texts, raw_outputs):
        try:
            # Parse
            if task == "ner":
                parsed = parser.parse(raw_output, input_item)
            else:
                parsed = parser.parse(raw_output)
            
            # Validate
            if task == "ner":
                validation_result = validate_ner_output(
                    parsed=parsed,
                    original_text=input_item,
                    valid_entities=entities,
                    format_type=output_format
                )

            results.append({
                'raw_output': raw_output,
                'parsed_output': parsed,
                'prediction': validation_result['prediction'],
                'fatal_error': validation_result['fatal_error'],
                'error_counts': validation_result['error_counts'],
                'prompt': prompt_text
            })
            
        except Exception as e:
            logger.error(f"Parsing/validation failed: {e}")
            # Count as malformed output
            if task == "ner":
                error_counts = {
                    'malformed_output': 1,
                    'length_mismatch': 0,
                    'invalid_label': 0,
                    'invalid_span': 0,
                    'matching_failed': 0
                }
                fallback = []

            results.append({
                'raw_output': raw_output,
                'parsed_output': None,
                'prediction': fallback,
                'fatal_error': 'malformed_output',
                'error_counts': error_counts,
                'prompt': prompt_text
            })
    
    return results


if __name__ == "__main__":
    task = "ner"
    model_name = "gpt-3.5-turbo"
    
    # Load model once
    llm = load_model(model_name)
    
    if task == "ner":
        input_format = "white_space"
        output_format = "iob"
        entities = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
        data_examples = [
            {
                "text": "Alex is going to Los Angeles",
                "entities": [
                    {"spans": [(0, 4)], "label": "PER", "text": "Alex"},
                    {"spans": [(17, 28)], "label": "LOC", "text": "Los Angeles"}
                ]
            },
            {
                "text": "Alex is going to Tokyo",
                "entities": [
                    {"spans": [(0, 4)], "label": "PER", "text": "Alex"},
                    {"spans": [(17, 22)], "label": "LOC", "text": "Tokyo"}
                ]
            }
        ]
        input_text = "Tokyo is where Alex is"

    result = predict(llm, input_format, output_format, entities, input_text, data_examples, task)
    
    logger.info("---Result---")
    logger.info(result)