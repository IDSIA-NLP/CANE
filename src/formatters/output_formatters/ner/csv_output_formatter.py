from formatters.output_formatters.base_output_formatter import BaseOutputFormatter

class CSVOutputFormatter(BaseOutputFormatter):
    def format(self, text, entities):
        """
        Format entities as comma-separated IOB tags (tags only, no tokens).

        Example output:
            "B-PER,O,O,B-LOC"
        """
        tokens, iob_tags = self._spans_to_iob_tags(text, entities)
        return ",".join(iob_tags)