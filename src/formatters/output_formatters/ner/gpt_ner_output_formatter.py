from formatters.output_formatters.base_output_formatter import BaseOutputFormatter

class GPTNEROutputFormatter(BaseOutputFormatter):
    def format(self, text, entities):
        """
        Format entities in GPT-NER style.

        Example: @@Turing##PER was born in @@London##LOC.
        Nested: @@myeloid @@PU.1##Protein binding site##DNA
        """
        if not entities:
            return text

        # Collect open/close events at character positions
        events = []
        for entity in entities:
            label = entity['label']
            # Use first span only (GPTNER doesn't support discontinuous)
            if entity['spans']:
                entity_start, entity_end = entity['spans'][0]
                span_len = entity_end - entity_start
                events.append((entity_start, 'open',  label, span_len))
                events.append((entity_end,   'close', label, span_len))

        # Sort events with tie-breaking:
        # - closes before opens at same position (adjacent entities)
        # - among closes: shorter span first (inner closes before outer)
        # - among opens: longer span first (outer opens before inner)
        def sort_key(event):
            pos, etype, label, span_len = event
            if etype == 'close':
                return (pos, 0,  span_len)   # closes first, shorter (inner) first
            else:
                return (pos, 1, -span_len)   # opens after, longer (outer) first

        events.sort(key=sort_key)

        # Build output by inserting annotations at character positions
        result = []
        last_pos = 0

        for pos, etype, label, span_len in events:
            result.append(text[last_pos:pos])
            last_pos = pos
            if etype == 'open':
                result.append('@@')
            else:
                result.append(f'##{label}')

        result.append(text[last_pos:])
        return ''.join(result)