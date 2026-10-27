from formatters.output_formatters.base_output_formatter import BaseOutputFormatter

class XMLRefOutputFormatter(BaseOutputFormatter):
    def format(self, text, entities):
        """
        Format entities as inline XML with ref attributes.
        Discontinuous and overlapping entities share ref ids across tags.

        Example: The <Disease ref="e1">heart</Disease> and <Disease ref="e2">chest</Disease> <Disease ref="e1 e2">pain</Disease>
        """
        if not entities:
            return text

        # Assign entity IDs and collect open/close events at character positions
        events = []
        for idx, entity in enumerate(entities):
            entity_id = f"e{idx + 1}"
            label = entity['label']

            # Process all spans (handles discontinuous)
            for span_start, span_end in entity['spans']:
                span_len = span_end - span_start
                events.append((span_start, 'open',  label, entity_id, span_len))
                events.append((span_end,   'close', label, entity_id, span_len))

        # Sort events with tie-breaking:
        # - closes before opens at same position (adjacent entities)
        # - among closes: shorter span first (inner closes before outer)
        # - among opens: longer span first (outer opens before inner)
        def sort_key(event):
            pos, etype, label, entity_id, span_len = event
            if etype == 'close':
                return (pos, 0,  span_len)   # closes first, shorter (inner) first
            else:
                return (pos, 1, -span_len)   # opens after, longer (outer) first

        events.sort(key=sort_key)

        # Build output by inserting tags at character positions
        result = []
        last_pos = 0

        for pos, etype, label, entity_id, span_len in events:
            result.append(text[last_pos:pos])
            last_pos = pos
            if etype == 'open':
                result.append(f'<{label} ref="{entity_id}">')
            else:
                result.append(f'</{label}>')

        result.append(text[last_pos:])
        return ''.join(result)