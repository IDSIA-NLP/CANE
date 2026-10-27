from formatters.output_formatters.base_output_formatter import BaseOutputFormatter

class PTFormatOutputFormatter(BaseOutputFormatter):
    def format(self, text, entities):
        """
        PT format from He and Choi (2023): double quotes, semicolons, no header.

        Example output:
            "cancer" is a Disease; "heart" is an Organ
        """
        if not entities:
            return "NONE"
        
        lines = []
        
        for entity in entities:
            entity_text = entity['text']
            entity_label = entity['label']
            
            # Format: "entity_text" is a LABEL
            lines.append(f'"{entity_text}" is a {entity_label}')
        
        # Join with "; "
        return "; ".join(lines)