from formatters.output_formatters.base_output_formatter import BaseOutputFormatter

class IOBOutputFormatter(BaseOutputFormatter):
    def format(self, text, entities):
        """
        Format entities as IOB tags with tokens, one per line.

        Example output:
            "Alex B-PER\\nwent O\\nto O\\nTokyo B-LOC"
        """
        tokens, iob_tags = self._spans_to_iob_tags(text, entities)
        
        ret = ""
        for token, label in zip(tokens, iob_tags):
            ret += f"{token} {label}\n"
        return ret.strip()  # Remove trailing newline