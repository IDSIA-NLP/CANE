from formatters.input_formatters.base_input_formatter import BaseInputFormatter

class WhitespaceInputFormatter(BaseInputFormatter):
    def format(self, text):
        """Format text as-is (whitespace-separated)."""
        return text