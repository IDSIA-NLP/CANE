from loguru import logger


def validate_ner_output(parsed, original_text, valid_entities, format_type):
    """
    Validate parser output for any NER format.

    parsed is either an error dict ({"error": ...}) or a list of entity dicts and
    error markers. Returns {'prediction', 'fatal_error', 'error_counts'}, where
    prediction holds only the entities that passed.
    """
    error_counts = {
        'malformed_output': 0,
        'length_mismatch': 0,
        'invalid_label': 0,
        'invalid_span': 0,
        'matching_failed': 0
    }
    
    # Parser returned an error
    if isinstance(parsed, dict) and 'error' in parsed:
        error_type = parsed['error']
        
        if error_type == 'length_mismatch':
            error_counts['length_mismatch'] = 1
            return {
                'prediction': [],
                'fatal_error': 'length_mismatch',
                'error_counts': error_counts
            }
        elif error_type == 'malformed':
            error_counts['malformed_output'] = 1
            return {
                'prediction': [],
                'fatal_error': 'malformed_output',
                'error_counts': error_counts
            }
        else:
            # Unknown error type, treat as malformed
            error_counts['malformed_output'] = 1
            return {
                'prediction': [],
                'fatal_error': 'malformed_output',
                'error_counts': error_counts
            }
    
    # Parsed output should be a list
    if not isinstance(parsed, list):
        error_counts['malformed_output'] = 1
        return {
            'prediction': [],
            'fatal_error': 'malformed_output',
            'error_counts': error_counts
        }
    
    # Validate entity content (invalid_label, invalid_span, matching_failed)
    validated_entities = []
    
    for entity in parsed:
        if not isinstance(entity, dict):
            error_counts['malformed_output'] = 1
            return {
                'prediction': [],
                'fatal_error': 'malformed_output',
                'error_counts': error_counts
            }
        
        # Error marker from the parser
        if 'error_type' in entity:
            error_type = entity['error_type']
            error_counts[error_type] += 1
            continue  # Skip this item, move to next
        
        # Check required fields
        if 'spans' not in entity or 'label' not in entity:
            error_counts['malformed_output'] = 1
            return {
                'prediction': [],
                'fatal_error': 'malformed_output',
                'error_counts': error_counts
            }
        
        label = entity['label']
        spans = entity['spans']
        
        # Invalid label
        label_valid = False
        if label in valid_entities:
            label_valid = True
        else:
            # Check if it's a valid entity type (removing B-/I- if present)
            base_label = label.split('-')[-1] if '-' in label else label
            if base_label in valid_entities:
                label_valid = True
            # Also check if valid_entities has B-/I- prefixes
            for valid_entity in valid_entities:
                if valid_entity.endswith(f'-{label}') or valid_entity.endswith(f'-{base_label}'):
                    label_valid = True
                    break
        
        if not label_valid:
            error_counts['invalid_label'] += 1
            continue  # Filter out this entity
        
        # Overlapping spans within the same entity
        sorted_spans = sorted(spans)
        has_overlap = False
        for i in range(len(sorted_spans) - 1):
            start1, end1 = sorted_spans[i]
            start2, end2 = sorted_spans[i + 1]
            # Check if spans overlap: (start1 < end2) and (start2 < end1)
            if start1 < end2 and start2 < end1:
                logger.warning(f"Overlapping spans within entity: {sorted_spans[i]} and {sorted_spans[i + 1]}")
                error_counts['malformed_output'] = 1
                return {
                    'prediction': [],
                    'fatal_error': 'malformed_output',
                    'error_counts': error_counts
                }
        
        # Invalid span (all formats produce span-based entities)
        valid_spans = []
        for start, end in spans:
            # Validate span bounds: start >= 0, end <= text length, start < end
            if start < 0 or end > len(original_text) or start >= end:
                error_counts['invalid_span'] += 1
                continue  # Skip this span
            valid_spans.append((start, end))
        
        # If all spans were invalid, filter out entity
        if not valid_spans:
            continue
        
        # Update entity with only valid spans
        entity = entity.copy()
        entity['spans'] = valid_spans
        
        # Entity passed all checks, keep it
        validated_entities.append(entity)
    
    return {
        'prediction': validated_entities,
        'fatal_error': None,
        'error_counts': error_counts
    }