from parsers.ner.base_ner_parser import BaseNERParser
from loguru import logger

class BRATParser(BaseNERParser):
    def parse(self, llm_output: str, original_text: str = None):
        """
        Parse BRAT standoff entity lines, e.g. "T1<TAB>Disease 0 6;10 15".

        Discontinuous spans are separated by ";". Entity text is cut from original_text.
        """
        llm_output = self.preprocess(llm_output)
        
        # Check for NONE marker
        none_check = self.check_for_none(llm_output)
        if none_check is not None:
            return none_check
        
        entities = []
        
        for line in llm_output.split("\n"):
            line = line.strip()
            if not line or not line.startswith('T'):
                continue
            
            parts = line.split("\t")
            if len(parts) != 2:
                logger.warning(f"Invalid BRAT line format (expected 2 tab-separated parts): {line}")
                return {"error": "malformed"}
            
            entity_id, type_and_spans = parts
            
            # Parse type and spans: "Disease 0 6;10 15"
            type_span_parts = type_and_spans.split(None, 1)
            if len(type_span_parts) != 2:
                logger.warning(f"Invalid type/spans format: {type_and_spans}")
                return {"error": "malformed"}
            
            entity_type, spans_str = type_span_parts
            
            # Parse spans: "0 6;10 15" -> [(0, 6), (10, 15)]
            try:
                spans = []
                for span_part in spans_str.split(";"):
                    span_nums = span_part.strip().split()
                    if len(span_nums) != 2:
                        logger.warning(f"Invalid span format: {span_part}")
                        return {"error": "malformed"}
                    spans.append((int(span_nums[0]), int(span_nums[1])))
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse spans '{spans_str}': {e}")
                return {"error": "malformed"}
            
            # Extract text from original_text using predicted spans
            entity_text = " ".join(
                original_text[start:end]
                for start, end in spans
                if original_text and start < len(original_text) and end <= len(original_text)
            )
            
            entities.append({
                'spans': spans,
                'label': entity_type,
                'text': entity_text
            })
        
        return entities