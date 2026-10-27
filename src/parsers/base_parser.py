from loguru import logger

class BaseParser:
    def preprocess(self, text: str) -> str:
        logger.debug(f"Raw LLM output: {text}")
        return text.strip().replace("\\n", "\n").replace("\\t", "\t")
    
    def parse(self, llm_output: str, original_text: str = None):
        """
        Parse LLM output into the internal format.

        original_text is only needed by formats that match entity text back to spans.
        """
        raise NotImplementedError("Subclasses should implement this method")