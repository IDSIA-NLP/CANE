from formatters.output_formatters.base_output_formatter import BaseOutputFormatter

class CANEBaseOutputFormatter(BaseOutputFormatter):
    def format(self, text, entities):
        """
        Format entities as `text` is a LABEL lines under an "Entities:" header.

        Example output:
            Entities:
            `cancer` is a Disease
            `heart` is an Organ
        """
        if not entities:
            return "NONE"
        
        lines = ["Entities:"]
        
        for entity in entities:
            entity_text = entity['text']
            entity_label = entity['label']
            
            # Format: `entity_text` is a LABEL
            lines.append(f'`{entity_text}` is a {entity_label}')
        
        # Join with newlines
        return "\n".join(lines)