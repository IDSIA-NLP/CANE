from formatters.output_formatters.base_output_formatter import BaseOutputFormatter

class CANEBaseAblHeaderOutputFormatter(BaseOutputFormatter):
    def format(self, text, entities):
        """
        cane_base ablation: no "Entities:" header.

        Example output:
            `cancer` is a Disease
            `heart` is an Organ
        """
        if not entities:
            return "NONE"
        
        lines = []
        
        for entity in entities:
            entity_text = entity['text']
            entity_label = entity['label']
            
            # Format: `entity_text` is a LABEL
            lines.append(f'`{entity_text}` is a {entity_label}')
        
        # Join with newlines
        return "\n".join(lines)