import re
from parsers.ner.base_ner_parser import BaseNERParser
from loguru import logger

class CANEDiscoParser(BaseNERParser):
    def parse(self, llm_output: str, original_text: str = None):
        """
        Parse CANE-Disco output, which supports discontinuous entities.

        Input format:
            Entities:
            `part1` and `part2` is a LABEL

        Parts that cannot be found in original_text give an error marker for that entity.
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
        
        # Strip leading "Entities:" if present
        if llm_output.strip().startswith("Entities:"):
            llm_output = llm_output.strip()[len("Entities:"):].strip()
        
        # Pattern: `entity_text` is a LABEL (or is an LABEL, or are a LABEL)
        label_pattern = r'(?:is|are)\s+an?\s+(\w+)'
        backtick_pattern = r'`([^`]+)`'
        
        # Parse LLM output into (parts, label) pairs
        llm_entities = []
        
        for line in llm_output.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            # Extract label
            label_match = re.search(label_pattern, line)
            
            if not label_match:
                logger.warning(f"Line doesn't match expected format: {line}")
                return {"error": "malformed"}
            
            entity_label = label_match.group(1)
            label_pos = label_match.start()
            
            # Extract all backtick-delimited parts before the label
            before_label = line[:label_pos]
            parts = [m.group(1).strip() for m in re.finditer(backtick_pattern, before_label)]
            
            if not parts:
                logger.warning(f"No backtick-delimited entities in: {line}")
                return {"error": "malformed"}
            
            if not entity_label:
                logger.warning(f"Empty label: {line}")
                return {"error": "malformed"}
            
            llm_entities.append((parts, entity_label))
        
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
        
        for parts, entity_label in llm_entities:
            # Match each part independently
            matched_spans = []
            match_failed = False
            
            for part in parts:
                # Find all possible positions for this part
                candidates = find_all_occurrences(part, original_text)
                
                # Pick first non-consumed position
                matched = False
                for candidate_span in candidates:
                    if not is_consumed(candidate_span):
                        matched_spans.append(candidate_span)
                        consumed_spans.append(candidate_span)
                        matched = True
                        break
                
                if not matched:
                    match_failed = True
                    break
            
            if match_failed:
                # Could not find one or more parts - add error marker
                results.append({
                    'error_type': 'matching_failed',
                    'entity_text': ' '.join(parts),
                    'label': entity_label
                })
            else:
                results.append({
                    'spans': matched_spans,
                    'label': entity_label,
                    'text': ' '.join(parts)
                })
        
        return results