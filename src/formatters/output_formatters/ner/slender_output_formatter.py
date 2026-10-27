from formatters.output_formatters.base_output_formatter import BaseOutputFormatter

class SLENDEROutputFormatter(BaseOutputFormatter):
    def format(self, text, entities):
        """
        Format entities as SLENDER: entity_text | LABEL, separated by semicolons.

        Example output:
            "Alex | PERSON; Tokyo | LOCATION"
        """
        if not entities:
            return "NONE"
        
        slender_parts = []
        
        for entity in entities:
            entity_text = entity['text']
            entity_label = entity['label']
            
            # Format: "entity_text | LABEL"
            slender_parts.append(f"{entity_text} | {entity_label}")
        
        # Join with semicolon separator
        return "; ".join(slender_parts)