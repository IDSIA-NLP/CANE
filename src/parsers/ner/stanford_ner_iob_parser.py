from parsers.ner.base_ner_parser import BaseNERParser
from loguru import logger

class StanfordNERIOBParser(BaseNERParser):
    def parse(self, llm_output: str, original_text: str = None):
        """
        Parse Stanford NER IOB format, e.g. "Alex/B-PERSON is/O going/O to/O Tokyo/B-LOCATION".
        """
        llm_output = self.preprocess(llm_output)
        
        # Extract tokens and IOB tags from LLM output
        tokens = []
        iob_tags = []
        
        for item in llm_output.split():
            if '/' not in item:
                logger.warning(f"Invalid Stanford NER IOB format: {item}")
                return {"error": "malformed"}
            
            parts = item.rsplit('/', 1)  # Split from right to handle tokens with '/'
            if len(parts) != 2:
                logger.warning(f"Invalid Stanford NER IOB token: {item}")
                return {"error": "malformed"}
            
            token, label = parts
            tokens.append(token)
            iob_tags.append(label)
        
        # Build token offsets from extracted tokens
        token_offsets = []
        char_offset = 0
        for token in tokens:
            start = char_offset
            end = char_offset + len(token)
            token_offsets.append((start, end))
            char_offset = end + 1  # +1 for space
        
        # Convert IOB tags to span-based entities
        entities = []
        current_entity = None
        
        for idx, (tag, (tok_start, tok_end)) in enumerate(zip(iob_tags, token_offsets)):
            token = tokens[idx]
            
            if tag == 'O':
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            elif tag.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                
                label = tag[2:]
                current_entity = {
                    'spans': [(tok_start, tok_end)],
                    'label': label,
                    'text': token
                }
            elif tag.startswith('I-'):
                label = tag[2:]
                if current_entity and current_entity['label'] == label:
                    current_entity['spans'][0] = (current_entity['spans'][0][0], tok_end)
                    current_entity['text'] += " " + token
                else:
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        'spans': [(tok_start, tok_end)],
                        'label': label,
                        'text': token
                    }
            else:
                logger.warning(f"Invalid IOB tag: {tag}")
                return {"error": "malformed"}
        
        if current_entity:
            entities.append(current_entity)
        
        return entities