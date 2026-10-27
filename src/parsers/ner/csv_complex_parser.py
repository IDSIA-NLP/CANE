import csv
from io import StringIO
from parsers.ner.base_ner_parser import BaseNERParser
from loguru import logger

class CSVComplexParser(BaseNERParser):
    def parse(self, llm_output: str, original_text: str = None):
        """
        Parse CSV lines with character offsets, e.g. "e1,Disease,0:6;10:15".

        Entity text is cut from original_text using the predicted spans.
        """
        llm_output = self.preprocess(llm_output)

        # Check for NONE marker
        none_check = self.check_for_none(llm_output)
        if none_check is not None:
            return none_check
        
        entities = []
        
        # Use CSV reader to handle quoted fields
        try:
            reader = csv.reader(StringIO(llm_output))
            for row in reader:
                if len(row) != 3:
                    logger.warning(f"Invalid CSV row format (expected 3 columns): {row}")
                    return {"error": "malformed"}
                
                entity_id, entity_type, spans_str = row
                
                # Parse spans: "0:6;10:15" -> [(0, 6), (10, 15)]
                try:
                    spans = []
                    for span_part in spans_str.split(";"):
                        start_str, end_str = span_part.split(":")
                        spans.append((int(start_str), int(end_str)))
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
        except csv.Error as e:
            logger.error(f"CSV parsing error: {e}")
            return {"error": "malformed"}
        
        return entities