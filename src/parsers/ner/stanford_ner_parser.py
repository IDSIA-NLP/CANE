from parsers.ner.base_ner_parser import BaseNERParser
from loguru import logger

class StanfordNERParser(BaseNERParser):
    def parse(self, llm_output: str, original_text: str = None):
        """Parse Stanford NER format, e.g. "Alex/PERSON is/O going/O to/O Tokyo/LOCATION"."""
        llm_output = self.preprocess(llm_output)
        
        # Extract tokens and labels from LLM output
        tokens = []
        labels = []
        
        for item in llm_output.split():
            if '/' not in item:
                logger.warning(f"Invalid Stanford NER format: {item}")
                return {"error": "malformed"}
            
            parts = item.rsplit('/', 1)  # Split from right to handle tokens with '/'
            if len(parts) != 2:
                logger.warning(f"Invalid Stanford NER token: {item}")
                return {"error": "malformed"}
            
            token, label = parts
            tokens.append(token)
            labels.append(label)
        
        # Build token offsets from extracted tokens
        token_offsets = []
        char_offset = 0
        for token in tokens:
            start = char_offset
            end = char_offset + len(token)
            token_offsets.append((start, end))
            char_offset = end + 1  # +1 for space
        
        # Convert to IOB tags (Stanford format has no B-/I- prefixes)
        iob_tags = []
        previous_tag = None
        
        for label in labels:
            if label != "O":
                iob_tag = f"B-{label}" if label != previous_tag else f"I-{label}"
            else:
                iob_tag = "O"
            previous_tag = label
            iob_tags.append(iob_tag)
        
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
        
        if current_entity:
            entities.append(current_entity)
        
        return entities