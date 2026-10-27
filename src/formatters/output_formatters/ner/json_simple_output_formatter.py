import json
from formatters.output_formatters.base_output_formatter import BaseOutputFormatter

class JSONSimpleOutputFormatter(BaseOutputFormatter):
    def format(self, text, entities):
        """
        Format entities as JSON grouped by label, without offsets.

        Example output:
            {
              "DRUG": ["amoxicillin"],
              "ADR": ["generalized rash"]
            }

        Each surface form appears once per label.
        """
        if not entities:
            return "NONE"
        
        # Group entity texts by label, preserving order and deduplicating
        label_to_entities = {}
        
        for entity in entities:
            label = entity['label']
            entity_text = entity['text']
            
            if label not in label_to_entities:
                label_to_entities[label] = []
            
            # Deduplicate surface forms per label
            if entity_text not in label_to_entities[label]:
                label_to_entities[label].append(entity_text)
        
        return json.dumps(label_to_entities, indent=2, ensure_ascii=False)