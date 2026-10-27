from formatters.output_formatters.base_output_formatter import BaseOutputFormatter

class CANEDiscoOutputFormatter(BaseOutputFormatter):
    def format(self, text, entities):
        """
        Like cane_base, but a discontinuous entity lists its parts joined by "and".

        Example output:
            Entities:
            `amoxicillin` is a DRUG
            `generalized rash` is a ADR
            `headache` and `nausea` are a ADR
        """
        if not entities:
            return "NONE"
        
        lines = ["Entities:"]
        
        for entity in entities:
            entity_label = entity['label']
            spans = sorted(entity['spans'])
            
            # Extract text for each span
            parts = [text[s[0]:s[1]] for s in spans]
            
            # Join parts with "and" for discontinuous entities
            quoted_parts = ' and '.join(f'`{p}`' for p in parts)
            
            # Use correct grammar: "are" for multiple parts, "is" for single
            verb = "are" if len(parts) > 1 else "is"
            
            # Format: `part1` and `part2` is/are a LABEL
            lines.append(f'{quoted_parts} {verb} a {entity_label}')
        
        # Join with newlines
        return "\n".join(lines)