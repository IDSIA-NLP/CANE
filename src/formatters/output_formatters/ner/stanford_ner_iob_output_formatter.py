from formatters.output_formatters.base_output_formatter import BaseOutputFormatter

class StanfordNerIOBOutputFormatter(BaseOutputFormatter):
    def format(self, text, entities):
        """
        Format entities as Stanford NER token/label pairs with IOB prefixes.

        Example output:
            "Alex/B-PERSON went/O to/O Tokyo/B-LOCATION"
        """
        tokens, iob_tags = self._spans_to_iob_tags(text, entities)
        
        formatted_output = []
        for token, label in zip(tokens, iob_tags):
            formatted_output.append(f"{token}/{label}")
        
        return " ".join(formatted_output)