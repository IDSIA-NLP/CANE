import json
from parsers.ner.base_ner_parser import BaseNERParser
from loguru import logger

class PubAnnotationParser(BaseNERParser):
    def parse(self, llm_output: str, original_text: str = None):
        """
        Parse PubAnnotation JSON. Each denotation becomes one entity.

        Input format:
            {"denotations": [{"id": "T1", "span": {"begin": 0, "end": 6}, "obj": "Disease"}]}
        """
        llm_output = self.preprocess(llm_output)
        
        try:
            data = json.loads(llm_output)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse PubAnnotation JSON: {e}")
            return {"error": "malformed"}
        
        if not isinstance(data, dict) or 'denotations' not in data:
            logger.error("Invalid PubAnnotation format: missing 'denotations'")
            return {"error": "malformed"}
        
        denotations = data['denotations']
        
        # Each denotation is treated as a separate entity
        entities = []
        
        for denotation in denotations:
            if not isinstance(denotation, dict):
                continue
            
            if 'span' not in denotation or 'obj' not in denotation:
                logger.warning(f"Invalid denotation format: {denotation}")
                continue
            
            span = denotation['span']
            if 'begin' not in span or 'end' not in span:
                logger.warning(f"Invalid span format: {span}")
                continue
            
            start = span['begin']
            end = span['end']
            entity_type = denotation['obj']
            
            # Extract text from original_text if available
            entity_text = original_text[start:end] if original_text and start < len(original_text) and end <= len(original_text) else ""
            
            entities.append({
                'spans': [(start, end)],
                'label': entity_type,
                'text': entity_text
            })
        
        return entities