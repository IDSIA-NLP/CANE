import re
from parsers.ner.base_ner_parser import BaseNERParser
from loguru import logger

class PubAnnotationInlineParser(BaseNERParser):
    def parse(self, llm_output: str, original_text: str = None):
        """
        Parse PubAnnotation Simple Inline Annotation format.

        Input format: "[Elon Musk][Person] is a member of the [PayPal Mafia][Organization]."
        Nested:       "[myeloid [PU.1][Protein] binding site][DNA]"

        A backslash before "[" escapes it. Nesting is handled with a stack.
        """
        llm_output = self.preprocess(llm_output)

        # Pattern matches either an unescaped '[' (open) or '][label]' (close)
        # Group 1: unescaped '[' (open marker)
        # Group 2: full '][label]' close marker
        # Group 3: label inside close marker
        token_pattern = re.compile(r'(?<!\\)(\[)|(\]\[([^\]]+)\])')

        entities = []

        # Stack-based parsing to support nested entities
        # Each stack entry: start_pos_in_clean_text
        stack = []
        clean_text_parts = []
        clean_pos = 0
        last_end = 0

        # Find all open/close markers in order
        for match in re.finditer(token_pattern, llm_output):
            is_open  = match.group(1) is not None
            is_close = match.group(2) is not None

            # Accumulate plain text between last token and this token
            # Remove escape characters from plain text
            text_between = llm_output[last_end:match.start()].replace('\\[', '[')
            clean_text_parts.append(text_between)
            clean_pos += len(text_between)
            last_end = match.end()

            if is_open:
                # Opening bracket: push current clean position onto stack
                stack.append(clean_pos)
            else:
                # Closing '][label]': pop innermost open bracket
                label = match.group(3).strip()

                # Skip empty labels
                if not label:
                    continue

                if not stack:
                    # No matching open bracket — skip this close marker
                    logger.warning(f"Unmatched close marker '][{label}]' — skipping")
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

        # Unclosed brackets remaining on the stack indicate malformed output
        if stack:
            logger.warning(f"Unclosed '[' markers remaining on stack: {len(stack)}")
            return {"error": "malformed"}

        return entities