import re
from parsers.ner.base_ner_parser import BaseNERParser
from loguru import logger

class XMLRefParser(BaseNERParser):
    def parse(self, llm_output: str, original_text: str = None):
        """
        Parse XML tags with ref attributes; tags sharing a ref form one discontinuous entity.

        Input format: '<Disease ref="e1">heart</Disease> ... <Disease ref="e1">pain</Disease>'
        """
        llm_output = self.preprocess(llm_output)

        # Pattern matches both opening tags with ref attributes and closing tags
        # Group 1: '/' for closing tag, '' for opening tag
        # Group 2: label
        # Group 3: ref value (only present for opening tags)
        tag_pattern = re.compile(r'<(/?)(\w+)(?:\s+ref="([^"]+)")?>')

        # Group spans by entity ref ID (handles discontinuous entities)
        entities_by_ref = {}

        # Stack-based parsing to support nested entities
        # Each stack entry: (label, ref_id, start_pos_in_clean_text)
        stack = []
        clean_text_parts = []
        clean_pos = 0
        last_end = 0

        # Find all XML tags in order
        for match in re.finditer(tag_pattern, llm_output):
            is_closing = match.group(1) == '/'
            label = match.group(2)
            ref   = match.group(3)

            # Skip 'O' or empty
            if label == 'O':
                last_end = match.end()
                continue

            # Accumulate plain text between last tag and this tag
            text_between = llm_output[last_end:match.start()]
            clean_text_parts.append(text_between)
            clean_pos += len(text_between)
            last_end = match.end()

            if not is_closing:
                # Opening tag: push label, ref and current clean position onto stack
                stack.append((label, ref, clean_pos))
            else:
                # Closing tag: find innermost matching label on stack
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i][0] == label:
                        _, ref_id, start_pos = stack.pop(i)
                        clean_text_so_far = ''.join(clean_text_parts)
                        entity_text = clean_text_so_far[start_pos:clean_pos]

                        # Skip empty entities
                        if not entity_text.strip():
                            break

                        # Group spans by (ref_id, label) to merge discontinuous spans
                        key = (ref_id, label)
                        if key not in entities_by_ref:
                            entities_by_ref[key] = {
                                'spans': [],
                                'label': label,
                                'text_parts': []
                            }
                        entities_by_ref[key]['spans'].append((start_pos, clean_pos))
                        entities_by_ref[key]['text_parts'].append(entity_text)
                        break

        # Unclosed tags remaining on the stack indicate malformed output
        if stack:
            logger.warning(f"Unclosed tags remaining on stack: {[s[0] for s in stack]}")
            return {"error": "malformed"}

        # Convert to entity list
        entities = []
        for (ref, label), entity_data in entities_by_ref.items():
            entities.append({
                'spans': entity_data['spans'],
                'label': label,
                'text': ' '.join(entity_data['text_parts'])
            })

        return entities