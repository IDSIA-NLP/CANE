from formatters.output_formatters.base_output_formatter import BaseOutputFormatter

class IOBSentenceOutputFormatter(BaseOutputFormatter):
    def format(self, text, entities):
        """
        Format entities as space-separated IOB tags (tags only, no tokens).

        Example output:
            "B-PER O O B-LOC"
        """
        tokens, iob_tags = self._spans_to_iob_tags(text, entities)
        return " ".join(iob_tags)