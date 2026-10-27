from parsers.base_parser import BaseParser

class BaseNERParser(BaseParser):
    def parse(self, llm_output: str, original_text: str = None):
        """Base parse method; subclasses implement the format-specific parsing."""
        return super().preprocess(llm_output)
    
    def check_for_none(self, llm_output: str):
        """Return [] if the output is the NONE marker (no entities), else None."""
        if llm_output.strip().upper() == "NONE":
            return []
        return None