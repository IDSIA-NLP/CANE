from formatters.output_formatters.base_output_formatter import BaseOutputFormatter

class StanfordNerOutputFormatter(BaseOutputFormatter):
    def format(self, text, entities):
        """
        Format entities as Stanford NER token/label pairs, without IOB prefixes.

        Example output:
            "Alex/PERSON went/O to/O Tokyo/LOCATION"
        """
        tokens, iob_tags = self._spans_to_iob_tags(text, entities)
        
        formatted_output = []
        for token, label in zip(tokens, iob_tags):
            # Remove B- and I- prefixes for Stanford format
            label = label.split("-")[-1]
            formatted_output.append(f"{token}/{label}")
        
        return " ".join(formatted_output)