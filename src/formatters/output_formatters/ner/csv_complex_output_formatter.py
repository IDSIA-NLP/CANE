from formatters.output_formatters.base_output_formatter import BaseOutputFormatter

class CSVComplexOutputFormatter(BaseOutputFormatter):
    def format(self, text, entities):
        """
        Format entities as CSV lines: entity_id,type,spans.
        Supports discontinuous and overlapping entities.

        Example output:
            "e1,Disease,0:6;10:15\\ne2,Symptom,0:20"
        """
        if not entities:
            return "NONE"
        
        lines = []
        
        for idx, entity in enumerate(entities):
            entity_id = f"e{idx + 1}"
            entity_type = entity['label']
            
            # Format spans: "start:end;start:end" for discontinuous
            span_strs = [f"{start}:{end}" for start, end in entity['spans']]
            spans_str = ";".join(span_strs)
            
            line = f"{entity_id},{entity_type},{spans_str}"
            lines.append(line)
        
        return "\n".join(lines)