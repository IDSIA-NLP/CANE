import json
from formatters.output_formatters.base_output_formatter import BaseOutputFormatter

class JSONOutputFormatter(BaseOutputFormatter):
    def format(self, text, entities):
        """
        Format entities as JSON with character spans.

        Example output:
            {
              "entities": [
                {
                  "label": "Symptom",
                  "spans": [{"start": 17, "end": 22}, {"start": 23, "end": 27}]
                }
              ]
            }
        """
        formatted_entities = []
        
        for entity in entities:
            formatted_entity = {
                "label": entity['label'],
                "spans": [{"start": start, "end": end} for start, end in entity['spans']]
            }
            formatted_entities.append(formatted_entity)
        
        result = {
            "entities": formatted_entities
        }
        
        return json.dumps(result, indent=2, ensure_ascii=False)