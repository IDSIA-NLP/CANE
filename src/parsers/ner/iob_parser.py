from parsers.ner.base_ner_parser import BaseNERParser
from loguru import logger

class IOBParser(BaseNERParser):
    def parse(self, llm_output: str, original_text: str = None):
        """
        Parse IOB lines ("token tag" per line).

        Tags are aligned to the whitespace tokens of original_text.
        """
        llm_output = self.preprocess(llm_output)
        
        if original_text is None:
            logger.error("original_text is required for tokenization")
            return {"error": "malformed"}
        
        # Extract IOB tags from LLM output
        iob_tags = []
        for line in llm_output.split("\n"):
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    iob_tags.append(parts[-1])  # Last part is the tag
                else:
                    logger.warning(f"Malformed line in IOB output: {line}")
                    return {"error": "malformed"}
        
        # Tokenize original text (whitespace only)
        tokens = original_text.split()
        
        # Check for length mismatch - this is a format error
        if len(tokens) != len(iob_tags):
            logger.warning(f"Length mismatch: {len(tokens)} tokens vs {len(iob_tags)} tags")
            return {"error": "length_mismatch"}
        
        # Build token character offsets from original text
        token_offsets = []
        char_offset = 0
        for token in tokens:
            start = original_text.find(token, char_offset)
            end = start + len(token)
            token_offsets.append((start, end))
            char_offset = end
        
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
                    # Extend current entity
                    current_entity['spans'][0] = (current_entity['spans'][0][0], tok_end)
                    current_entity['text'] += " " + token
                else:
                    # Start new entity (malformed IOB but we handle it)
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        'spans': [(tok_start, tok_end)],
                        'label': label,
                        'text': token
                    }
            else:
                # Invalid tag format
                logger.warning(f"Invalid tag format: {tag}")
                return {"error": "malformed"}
        
        if current_entity:
            entities.append(current_entity)
        
        return entities