from formatters.output_formatters.base_output_formatter import BaseOutputFormatter

class CANEBaseAblNewlineOutputFormatter(BaseOutputFormatter):
    def format(self, text, entities):
        """
        cane_base ablation: entities separated by semicolons instead of newlines.

        Example output:
            Entities:
            `cancer` is a Disease; `heart` is an Organ
        """
        if not entities:
            return "NONE"
        
        lines = ["Entities:"]
        
        for entity in entities:
            entity_text = entity['text']
            entity_label = entity['label']
            
            # Format: `entity_text` is a LABEL
            lines.append(f'`{entity_text}` is a {entity_label}')
        
        # Join header with newline, join entity annotations with "; "
        return lines[0] + "\n" + "; ".join(lines[1:])