from collections import defaultdict
from formatters.output_formatters.base_output_formatter import BaseOutputFormatter

class SymbolicTaggingOutputFormatter(BaseOutputFormatter):
    def format(self, text, entities):
        """
        Format entities in symbolic tagging style: one line per entity type,
        repeating the text with [ ] around spans of that type.

        Example:
            protein:   [CD28] activates T lymphocytes.
            cell_type: CD28 activates [T lymphocytes].
        Nested same-type entities get nested brackets:
            protein:   Activation of the [[CD28] surface receptor] ...

        Newlines become spaces so each type stays on one line. Offsets are
        unchanged since the substitution is 1-to-1.
        """
        if not entities:
            return text.replace('\n', ' ')

        # Normalize newlines to spaces (preserves character offsets, 1-to-1)
        normalized_text = text.replace('\n', ' ')

        # Group entities by label
        entities_by_label = defaultdict(list)
        for entity in entities:
            entities_by_label[entity['label']].append(entity)

        lines = []
        for label, label_entities in entities_by_label.items():
            # Collect open/close bracket events at character positions
            events = []
            for entity in label_entities:
                # Use first span only (symbolic tagging doesn't support discontinuous)
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
                pos, etype, lbl, span_len = event
                if etype == 'close':
                    return (pos, 0,  span_len)   # closes first, shorter (inner) first
                else:
                    return (pos, 1, -span_len)   # opens after, longer (outer) first

            events.sort(key=sort_key)

            # Build annotated sentence for this label
            result = []
            last_pos = 0

            for pos, etype, lbl, span_len in events:
                result.append(normalized_text[last_pos:pos])
                last_pos = pos
                if etype == 'open':
                    result.append('[')
                else:
                    result.append(']')

            result.append(normalized_text[last_pos:])
            lines.append(f"{label}: {''.join(result)}")

        return '\n'.join(lines)