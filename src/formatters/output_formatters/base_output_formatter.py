class BaseOutputFormatter:
    def format(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method")
    
    def _spans_to_iob_tags(self, text, entities):
        """
        Convert span entities to IOB tags over whitespace tokens.

        Only the first span of each entity is used. Returns (tokens, iob_tags).
        """
        tokens = text.split()
        iob_tags = ['O'] * len(tokens)
        
        # Build token offsets
        token_offsets = []
        char_offset = 0
        for token in tokens:
            start = text.find(token, char_offset)
            end = start + len(token)
            token_offsets.append((start, end))
            char_offset = end
        
        # Map entities to tokens (use first span only for simple formats)
        for entity in entities:
            label = entity['label']
            if entity['spans']:
                entity_start, entity_end = entity['spans'][0]
                
                first_token_idx = None
                last_token_idx = None
                
                for idx, (tok_start, tok_end) in enumerate(token_offsets):
                    if tok_start < entity_end and tok_end > entity_start:
                        if first_token_idx is None:
                            first_token_idx = idx
                        last_token_idx = idx
                
                if first_token_idx is not None:
                    iob_tags[first_token_idx] = f'B-{label}'
                    for idx in range(first_token_idx + 1, last_token_idx + 1):
                        iob_tags[idx] = f'I-{label}'
        
        return tokens, iob_tags