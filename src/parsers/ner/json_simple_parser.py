import json
from parsers.ner.base_ner_parser import BaseNERParser
from loguru import logger

class JSONSimpleParser(BaseNERParser):
    def parse(self, llm_output: str, original_text: str = None):
        """
        Parse label-grouped entity strings without offsets.

        Input format: {"DRUG": ["amoxicillin"], "ADR": ["generalized rash"]}

        Spans are recovered by string matching in original_text.
        """
        llm_output = self.preprocess(llm_output)

        # Check for NONE marker
        none_check = self.check_for_none(llm_output)
        if none_check is not None:
            return none_check

        if original_text is None:
            logger.error("original_text is required for span reconstruction")
            return {"error": "malformed"}

        try:
            data = json.loads(llm_output)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return {"error": "malformed"}

        # Validate structure
        if not isinstance(data, dict):
            logger.error("JSON output is not a dictionary")
            return {"error": "malformed"}

        for label, entity_list in data.items():
            if not isinstance(entity_list, list):
                logger.warning(f"Value for label '{label}' is not a list")
                return {"error": "malformed"}

        # Find all occurrences of entity_text in original_text (case-sensitive, left-to-right)
        def find_all_occurrences(entity_text, text):
            """Find all positions where entity_text appears (case-sensitive)."""
            positions = []
            search_pos = 0

            while search_pos < len(text):
                pos = text.find(entity_text, search_pos)
                if pos == -1:
                    break
                positions.append((pos, pos + len(entity_text)))
                search_pos = pos + 1

            return positions

        # Track consumed spans
        consumed_spans = []

        def is_consumed(span):
            """Check if span overlaps with any consumed span."""
            start, end = span
            for c_start, c_end in consumed_spans:
                if start < c_end and end > c_start:
                    return True
            return False

        # Match all occurrences of each entity string per label
        results = []

        for label, entity_list in data.items():
            for entity_text in entity_list:
                if not isinstance(entity_text, str) or not entity_text:
                    logger.warning(f"Invalid entity text for label '{label}': {entity_text}")
                    return {"error": "malformed"}

                candidates = find_all_occurrences(entity_text, original_text)

                # Add all non-consumed occurrences
                matched = False
                for candidate_span in candidates:
                    if not is_consumed(candidate_span):
                        results.append({
                            'spans': [candidate_span],
                            'label': label,
                            'text': entity_text
                        })
                        consumed_spans.append(candidate_span)
                        matched = True

                if not matched:
                    # Could not find entity - add error marker
                    results.append({
                        'error_type': 'matching_failed',
                        'entity_text': entity_text,
                        'label': label
                    })

        return results