import re
from parsers.ner.base_ner_parser import BaseNERParser
from loguru import logger

class SymbolicTaggingParser(BaseNERParser):
    def parse(self, llm_output: str, original_text: str = None):
        """
        Parse symbolic tagging: one copy of the sentence per entity type.

        Input format:
            protein:   [CD28] activates T lymphocytes.
            cell_type: CD28 activates [T lymphocytes].
        Nested:
            protein:   Activation of the [[CD28] surface receptor] ...
        """
        llm_output = self.preprocess(llm_output)

        # Pattern matches either '[' (open) or ']' (close)
        token_pattern = re.compile(r'(\[)|(\])')

        entities = []

        for line in llm_output.splitlines():
            line = line.strip()
            if not line:
                continue

            # Extract label and annotated sentence from "label: sentence"
            colon_idx = line.find(':')
            if colon_idx == -1:
                logger.warning(f"Skipping line without label prefix: {line!r}")
                continue

            label = line[:colon_idx].strip()
            annotated_sentence = line[colon_idx + 1:].strip()

            if not label or not annotated_sentence:
                continue

            # Stack-based parsing to support nested same-type entities
            # Each stack entry: start_pos_in_clean_text
            stack = []
            clean_text_parts = []
            clean_pos = 0
            last_end = 0

            # Find all open/close bracket markers in order
            for match in re.finditer(token_pattern, annotated_sentence):
                is_open  = match.group(1) is not None
                is_close = match.group(2) is not None

                # Accumulate plain text between last token and this token
                text_between = annotated_sentence[last_end:match.start()]
                clean_text_parts.append(text_between)
                clean_pos += len(text_between)
                last_end = match.end()

                if is_open:
                    # Opening [: push current clean position onto stack
                    stack.append(clean_pos)
                else:
                    # Closing ]: pop innermost open bracket
                    if not stack:
                        logger.warning(f"Unmatched ']' in label '{label}' — skipping")
                        continue

                    start_pos = stack.pop()
                    clean_text_so_far = ''.join(clean_text_parts)
                    entity_text = clean_text_so_far[start_pos:clean_pos]

                    # Skip empty entities
                    if not entity_text.strip():
                        continue

                    entities.append({
                        'spans': [(start_pos, clean_pos)],
                        'label': label,
                        'text': entity_text
                    })

        return entities