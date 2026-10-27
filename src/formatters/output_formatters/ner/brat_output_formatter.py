from formatters.output_formatters.base_output_formatter import BaseOutputFormatter

class BRATOutputFormatter(BaseOutputFormatter):
    def format(self, text, entities):
        """
        Format entities as BRAT standoff (entities only, no relations).

        Example output:
            "T1\\tDisease 0 6;10 15\\nT2\\tSymptom 0 20"

        Discontinuous spans are joined with semicolons.
        """
        if not entities:
            return "NONE"
        
        lines = []
        
        for idx, entity in enumerate(entities):
            entity_id = f"T{idx + 1}"
            entity_type = entity['label']
            
            # Format spans: "start end;start end" for discontinuous
            span_strs = [f"{start} {end}" for start, end in entity['spans']]
            spans_str = ";".join(span_strs)
            
            line = f"{entity_id}\t{entity_type} {spans_str}"
            lines.append(line)
        
        return "\n".join(lines)