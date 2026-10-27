import re
from parsers.ner.base_ner_parser import BaseNERParser
from loguru import logger

class XMLSimpleParser(BaseNERParser):
    def parse(self, llm_output: str, original_text: str = None):
        """
        Parse inline XML tags, e.g. "The <Disease>cancer</Disease> spread". Nesting is allowed.
        """
        llm_output = self.preprocess(llm_output)

        # Remove all XML tags to get plain text
        text_without_tags = re.sub(r'<[^>]+>', '', llm_output)

        entities = []

        # Stack-based parsing to support nested entities
        # Each stack entry: (label, start_pos_in_clean_text)
        stack = []
        clean_text_parts = []
        clean_pos = 0
        last_end = 0

        tag_pattern = re.compile(r'<(/?)(\w+)>')

        for match in re.finditer(tag_pattern, llm_output):
            is_closing = match.group(1) == '/'
            label = match.group(2)

            # Skip 'O' labels
            if label == 'O':
                last_end = match.end()
                continue

            # Accumulate plain text between last tag and this tag
            text_between = llm_output[last_end:match.start()]
            clean_text_parts.append(text_between)
            clean_pos += len(text_between)
            last_end = match.end()

            if not is_closing:
                # Opening tag: push current clean position onto stack
                stack.append((label, clean_pos))
            else:
                # Closing tag: find innermost matching label on stack
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i][0] == label:
                        _, start_pos = stack.pop(i)
                        clean_text_so_far = ''.join(clean_text_parts)
                        entity_text = clean_text_so_far[start_pos:clean_pos]

                        # Skip empty entities
                        if not entity_text.strip():
                            break

                        entities.append({
                            'spans': [(start_pos, clean_pos)],
                            'label': label,
                            'text': entity_text
                        })
                        break

        # Unclosed tags remaining on the stack indicate malformed output
        if stack:
            logger.warning(f"Unclosed tags remaining on stack: {[s[0] for s in stack]}")
            return {"error": "malformed"}

        return entities