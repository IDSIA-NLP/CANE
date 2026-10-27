import json
from parsers.ner.base_ner_parser import BaseNERParser
from loguru import logger

class JSONParser(BaseNERParser):
    def parse(self, llm_output: str, original_text: str = None):
        """
        Parse JSON entities with character offsets (discontinuous and overlapping allowed).

        Input format:
            {"entities": [{"label": "Symptom",
                           "spans": [{"start": 17, "end": 22}, {"start": 23, "end": 27}]}]}

        Entity text is cut from original_text.
        """
        llm_output = self.preprocess(llm_output)
        
        try:
            data = json.loads(llm_output)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return {"error": "malformed"}
        
        # Validate structure
        if not isinstance(data, dict):
            logger.error("JSON output is not a dictionary")
            return {"error": "malformed"}
        
        if 'entities' not in data:
            logger.error("JSON missing 'entities' field")
            return {"error": "malformed"}
        
        if not isinstance(data['entities'], list):
            logger.error("'entities' field is not a list")
            return {"error": "malformed"}
        
        # Extract entities
        entities = []
        
        for entity in data['entities']:
            if not isinstance(entity, dict):
                logger.warning(f"Invalid entity format (not a dict): {entity}")
                return {"error": "malformed"}
            
            # Validate required fields
            if 'label' not in entity or 'spans' not in entity:
                logger.warning(f"Entity missing 'label' or 'spans': {entity}")
                return {"error": "malformed"}
            
            label = entity['label']
            spans_list = entity['spans']
            
            # Validate spans
            if not isinstance(spans_list, list):
                logger.warning(f"Spans is not a list: {spans_list}")
                return {"error": "malformed"}
            
            spans = []
            for span in spans_list:
                if not isinstance(span, dict):
                    logger.warning(f"Invalid span format (not a dict): {span}")
                    return {"error": "malformed"}
                
                if 'start' not in span or 'end' not in span:
                    logger.warning(f"Span missing 'start' or 'end': {span}")
                    return {"error": "malformed"}
                
                try:
                    start = int(span['start'])
                    end = int(span['end'])
                    spans.append((start, end))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid span values: {span}, error: {e}")
                    return {"error": "malformed"}
            
            if not spans:
                logger.warning("Entity has no valid spans")
                return {"error": "malformed"}
            
            # Extract text from original_text using predicted spans
            entity_text = " ".join(
                original_text[start:end]
                for start, end in spans
                if original_text and start < len(original_text) and end <= len(original_text)
            )
            
            entities.append({
                'spans': spans,
                'label': label,
                'text': entity_text
            })
        
        return entities