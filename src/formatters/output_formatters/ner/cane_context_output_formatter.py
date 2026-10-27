from formatters.output_formatters.base_output_formatter import BaseOutputFormatter


class CANEContextOutputFormatter(BaseOutputFormatter):
    """
    CANE-Context: CANE-Full without discontinuous entities.

    Format: Entities:\n`entity_text` is a LABEL found in `left entity right`
    Only the first span of an entity is used.

    n is the context size, in whitespace-separated units, on each side.
    """
    
    def __init__(self, n=1):
        self.n = n
    
    def format(self, text, entities):
        """One line per entity under an "Entities:" header."""
        if not entities:
            return "NONE"
        
        lines = ["Entities:"]
        for entity in entities:
            line = self._format_entity(text, entity)
            lines.append(line)
        
        return "\n".join(lines)
    
    def _format_entity(self, text, entity):
        """Format a single continuous entity."""
        label = entity['label']
        spans = sorted(entity['spans'])
        
        first_start = spans[0][0]
        last_end = spans[0][1]
        left_context = self._get_left_context(text, first_start)
        right_context = self._get_right_context(text, last_end)
        entity_text = text[spans[0][0]:spans[0][1]]
        return self._format_continuous(label, entity_text, left_context, right_context)
    
    def _format_continuous(self, label, entity_text, left_ctx, right_ctx):
        """Format a continuous (single-span) entity."""
        verbatim_context = left_ctx + entity_text + right_ctx
        return f'`{entity_text}` is a {label} found in `{verbatim_context}`'
    
    # Context extraction helpers
    
    def _get_left_context(self, text, entity_start):
        """Get left context with n context elements."""
        # Find the start of the whitespace-unit containing entity_start
        unit_start = entity_start
        while unit_start > 0 and not text[unit_start - 1].isspace():
            unit_start -= 1
        
        # Check if there's a prefix
        prefix = text[unit_start:entity_start] if unit_start < entity_start else ""
        
        if prefix:
            if self.n == 1:
                return prefix
            else:
                pos = unit_start
                while pos > 0 and text[pos - 1].isspace():
                    pos -= 1
                
                if pos == 0:
                    return prefix
                
                additional = self._collect_n_units_left(text, pos, self.n - 1)
                return additional + prefix
        else:
            pos = entity_start
            while pos > 0 and text[pos - 1].isspace():
                pos -= 1
            
            if pos == 0:
                return ""
            
            return self._collect_n_units_left(text, pos, self.n)
    
    def _get_right_context(self, text, entity_end):
        """Get right context with n context elements."""
        # Find the end of the whitespace-unit containing entity_end
        unit_end = entity_end
        while unit_end < len(text) and not text[unit_end].isspace():
            unit_end += 1
        
        # Check if there's a suffix
        suffix = text[entity_end:unit_end] if entity_end < unit_end else ""
        
        if suffix:
            if self.n == 1:
                return suffix
            else:
                pos = unit_end
                while pos < len(text) and text[pos].isspace():
                    pos += 1
                
                if pos >= len(text):
                    return suffix
                
                additional = self._collect_n_units_right(text, pos, self.n - 1)
                return suffix + additional
        else:
            pos = entity_end
            while pos < len(text) and text[pos].isspace():
                pos += 1
            
            if pos >= len(text):
                return ""
            
            return self._collect_n_units_right(text, pos, self.n)
    
    def _collect_n_units_left(self, text, pos, n):
        """Collect n whitespace-units going left from position pos."""
        units = []
        current_pos = pos
        
        for _ in range(n):
            if current_pos == 0:
                break
            
            unit_end = current_pos
            while current_pos > 0 and not text[current_pos - 1].isspace():
                current_pos -= 1
            
            unit = text[current_pos:unit_end]
            units.append(unit)
            
            while current_pos > 0 and text[current_pos - 1].isspace():
                current_pos -= 1
        
        return " ".join(reversed(units)) + " " if units else ""
    
    def _collect_n_units_right(self, text, pos, n):
        """Collect n whitespace-units going right from position pos."""
        units = []
        current_pos = pos
        
        for _ in range(n):
            if current_pos >= len(text):
                break
            
            unit_start = current_pos
            while current_pos < len(text) and not text[current_pos].isspace():
                current_pos += 1
            
            unit = text[unit_start:current_pos]
            units.append(unit)
            
            while current_pos < len(text) and text[current_pos].isspace():
                current_pos += 1
        
        return " " + " ".join(units) if units else ""