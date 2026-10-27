from formatters.formatter_factory import FormatterFactory
from formatters.format_guidelines import FORMAT_GUIDELINES

GENERAL_OUTPUT_RULE = ("Output only the final answer in the required format. "
                       "All predictions must exactly match the input verbatim. "
                       "Do not alter capitalization, do not add or remove whitespace, and do not change punctuation. "
                       "Do not add explanations, commentary, markdown, or any extra text.")

def build_ner_context(input_format, output_format, entities, data_examples):
    """Build NER prompt context with format guidelines and examples."""
    

    # Format entity list as comma-separated string
    entity_list = ", ".join(sorted(entities))
    
    # Get format guideline
    guideline = FORMAT_GUIDELINES.get(output_format, "Follow the format shown in the examples.")
    
    # Build context
    context = f"Task: Named Entity Recognition\n\n"
    context += f"Entities: {entity_list}\n\n"
    context += f"Format: {output_format}\n"
    context += f"Guidelines: {guideline}\n\n"
    
    context += f"Important: {GENERAL_OUTPUT_RULE}\n\n"
    
    context += "Examples:\n"
    
    # Get formatters
    input_formatter = FormatterFactory.get_formatter(input_format, 'ner', 'input')
    output_formatter = FormatterFactory.get_formatter(output_format, 'ner', 'output')
    
    # Add examples
    for n, example in enumerate(data_examples):
        context += f"\nInput: {input_formatter.format(example['text'])}\n"
        example_output = output_formatter.format(example["text"], example["entities"]).strip("\n")
        context += f"Output:\n{example_output}\n"
    
    return context