import json
from formatters.output_formatters.base_output_formatter import BaseOutputFormatter

class PubAnnotationOutputFormatter(BaseOutputFormatter):
    def format(self, text, entities):
        """
        Format entities as PubAnnotation JSON.

        Example output:
            {
              "denotations": [
                {"id": "T1", "span": {"begin": 17, "end": 21}, "obj": "Symptom"}
              ]
            }

        A discontinuous entity becomes one denotation per span.
        """
        denotations = []
        entity_counter = 1
        
        for entity in entities:
            entity_type = entity['label']
            
            # For discontinuous entities, create one denotation per span
            for span_start, span_end in entity['spans']:
                denotations.append({
                    "id": f"T{entity_counter}",
                    "span": {"begin": span_start, "end": span_end},
                    "obj": entity_type
                })
                entity_counter += 1
        
        result = {
            "denotations": denotations
        }
        
        return json.dumps(result, indent=2)