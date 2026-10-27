from formatters.output_formatters.base_output_formatter import BaseOutputFormatter

class XMLSimpleOutputFormatter(BaseOutputFormatter):
    def format(self, text, entities):
        """
        Format entities as simple inline XML without refs.

        Example: The <Disease>cancer</Disease> spread to <Organ>lung</Organ>
        Nested: <DNA>myeloid <Protein>PU.1</Protein> binding site</DNA>
        """
        if not entities:
            return text

        # Collect open/close tag events at character positions
        events = []
        for entity in entities:
            label = entity['label']
            # Use first span only (simple XML doesn't support discontinuous)
            if entity['spans']:
                start, end = entity['spans'][0]
                span_len = end - start
                events.append((start, 'open',  label, span_len))
                events.append((end,   'close', label, span_len))

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

        # Build output by inserting tags at character positions
        result = []
        last_pos = 0

        for pos, etype, label, span_len in events:
            result.append(text[last_pos:pos])
            last_pos = pos
            if etype == 'open':
                result.append(f'<{label}>')
            else:
                result.append(f'</{label}>')

        result.append(text[last_pos:])
        return ''.join(result)