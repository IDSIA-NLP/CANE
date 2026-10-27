import re
from parsers.ner.base_ner_parser import BaseNERParser
from loguru import logger

class PTFormatParser(BaseNERParser):
    def parse(self, llm_output: str, original_text: str = None):
        """
        Parse the PT format from He & Choi (2023).

        Input format: "entity_text" is a LABEL; "entity_text" is a LABEL

        Entities are matched back to original_text. Ones that can't be found
        become error markers in the result list.
        """
        llm_output = self.preprocess(llm_output)

        # Check for NONE marker
        none_check = self.check_for_none(llm_output)
        if none_check is not None:
            return none_check
        
        if original_text is None:
            logger.error("original_text is required for span reconstruction")
            return {"error": "malformed"}
        
        # Handle empty output
        if not llm_output.strip():
            return []
        
        # Pattern: "entity_text" is a LABEL (or is an LABEL)
        pattern = r'"([^"]+)"\s+is\s+an?\s+(\w+)'
        
        # Parse LLM output into (entity_text, label) pairs
        llm_entities = []
        
        for part in llm_output.split(";"):
            part = part.strip()
            if not part:
                continue
            
            # Match the pattern
            match = re.match(pattern, part)
            
            if not match:
                logger.warning(f"Part doesn't match expected format: {part}")
                return {"error": "malformed"}
            
            entity_text = match.group(1).strip() 
            entity_label = match.group(2)
            
            if not entity_text or not entity_label:
                logger.warning(f"Empty entity text or label: {part}")
                return {"error": "malformed"}
            
            llm_entities.append((entity_text, entity_label))
        
        # Find all occurrences of each entity text (case-sensitive)
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
        
        # Match entities order-independently
        results = []
        
        for entity_text, entity_label in llm_entities:
            # Find all possible positions
            candidates = find_all_occurrences(entity_text, original_text)
            
            # Pick first non-consumed position
            matched = False
            for candidate_span in candidates:
                if not is_consumed(candidate_span):
                    # Successfully matched
                    results.append({
                        'spans': [candidate_span],
                        'label': entity_label,
                        'text': entity_text
                    })
                    consumed_spans.append(candidate_span)
                    matched = True
                    break
            
            if not matched:
                # Could not find entity - add error marker
                results.append({
                    'error_type': 'matching_failed',
                    'entity_text': entity_text,
                    'label': entity_label
                })
        
        return results