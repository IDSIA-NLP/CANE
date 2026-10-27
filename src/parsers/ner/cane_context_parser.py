import re
from parsers.ner.base_ner_parser import BaseNERParser
from loguru import logger


class CANEContextParser(BaseNERParser):
    """
    CANE-Context: each entity comes with a context string used to locate it.
    No discontinuous entities. Context matching tolerates whitespace differences.
    """
    
    def parse(self, llm_output: str, original_text: str = None):
        """
        Parse CANE-Context output.

        Input format:
            Entities:
            `entity` is a LABEL found in `context`

        Entities that cannot be located come back as error markers.
        """
        llm_output = self.preprocess(llm_output)

        # Check for NONE marker
        none_check = self.check_for_none(llm_output)
        if none_check is not None:
            return none_check
        
        if original_text is None:
            logger.error("original_text is required for CANE-Context parser")
            return {"error": "malformed"}
        
        if not llm_output.strip():
            return []

        # Strip leading "Entities:" if present
        if llm_output.strip().startswith("Entities:"):
            llm_output = llm_output.strip()[len("Entities:"):].strip()
        
        try:
            return self._parse_natural(llm_output, original_text)
        except Exception as e:
            logger.error(f"Parse error in CANE-Context format: {e}")
            return {"error": "malformed"}
    
    def _parse_natural(self, llm_output, original_text):
        """Parse CANE-Context format with order-independent matching."""
        llm_entities = []
        
        for line in llm_output.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            # Extract label: "is a LABEL found in" or "is an LABEL found in"
            label_pattern = r'(?:is)\s+an?\s+(\w+)\s+found\s+in'
            label_match = re.search(label_pattern, line)
            
            if not label_match:
                logger.warning(f"No label found in: {line}")
                continue
            
            label = label_match.group(1)
            label_pos = label_match.start()
            
            # Extract backtick-delimited entity text before the label
            before_label = line[:label_pos]
            
            # Find backtick-delimited strings
            backtick_pattern = r'`([^`]*)`'
            quoted_matches = list(re.finditer(backtick_pattern, before_label))
            
            if not quoted_matches:
                logger.warning(f"No backtick-delimited entities in: {line}")
                continue
            
            # Extract single entity part
            part = quoted_matches[0].group(1).strip()
            
            # Extract single backtick-delimited context string from 'found in `context`'
            after_label = line[label_match.end():].strip()
            context_matches = list(re.finditer(backtick_pattern, after_label))
            
            if not context_matches:
                logger.warning(f"No context found in: {line}")
                continue
            
            # Extract single context string
            context = context_matches[0].group(1)
            
            # Build entity context from the single part and context
            ctx_info = self._extract_entity_context_from_string(context, part)
            if ctx_info is None:
                logger.warning(f"Could not find entity '{part}' in context '{context}'")
                continue
            
            llm_entities.append({
                'label': label,
                'parts': [part],
                'entity_contexts': [ctx_info],
                'entity_text': part
            })
        
        # Match entities order-independently
        consumed_spans = []
        results = []
        
        for entity_info in llm_entities:
            spans = self._find_entity_parts_with_contexts(
                original_text,
                entity_info['entity_contexts'],
                consumed_spans
            )
            
            if spans is None:
                results.append({
                    'error_type': 'matching_failed',
                    'entity_text': entity_info['entity_text'],
                    'label': entity_info['label']
                })
            else:
                results.append({
                    'spans': spans,
                    'label': entity_info['label'],
                    'text': entity_info['entity_text']
                })
                consumed_spans.extend(spans)
        
        return results
    
    def _extract_entity_context_from_string(self, context_string, entity_part):
        """
        Split a context string into left/right context around the entity part.

        Returns {'entity': part, 'left': left_ctx, 'right': right_ctx}.
        """
        # Find the entity part in the context string
        part_pos = context_string.find(entity_part)
        
        if part_pos == -1:
            return None
        
        # Extract left and right context
        left_ctx = context_string[:part_pos]
        right_ctx = context_string[part_pos + len(entity_part):]
        
        return {
            'entity': entity_part,
            'left': left_ctx,
            'right': right_ctx
        }
    
    def _find_entity_parts_with_contexts(self, original_text, entity_contexts, consumed_spans):
        """Find all entity parts using their contexts."""
        spans = []
        search_start = 0
        
        for ctx_info in entity_contexts:
            entity_part = ctx_info['entity']
            left_ctx = ctx_info['left']
            right_ctx = ctx_info['right']
            
            # Find this part with whitespace-flexible matching
            span = self._find_entity_with_context_flexible(
                original_text[search_start:],
                left_ctx,
                entity_part,
                right_ctx,
                consumed_spans
            )
            
            if span is None:
                return None
            
            # Adjust span to absolute position
            abs_span = (search_start + span[0], search_start + span[1])
            
            # Check if consumed
            if self._is_consumed(abs_span, consumed_spans):
                return None
            
            spans.append(abs_span)
            search_start = abs_span[1]
        
        return spans
    
    def _find_entity_with_context_flexible(self, text, left_ctx, entity_text, right_ctx, consumed_spans):
        """
        Find the entity via its context, allowing whitespace differences.

        Returns (start, end) or None.
        """
        if not entity_text:
            return None
        
        # Build flexible pattern that allows extra whitespace
        # Normalize whitespace in context parts
        left_ctx = left_ctx.strip()
        entity_text = entity_text.strip()
        right_ctx = right_ctx.strip()
        
        # Escape special regex characters
        entity_escaped = re.escape(entity_text)
        
        # Build pattern with optional whitespace flexibility
        if left_ctx and right_ctx:
            left_escaped = re.escape(left_ctx)
            right_escaped = re.escape(right_ctx)
            # Allow optional whitespace around entity
            pattern = left_escaped + r'\s*' + entity_escaped + r'\s*' + right_escaped
        elif left_ctx:
            left_escaped = re.escape(left_ctx)
            pattern = left_escaped + r'\s*' + entity_escaped
        elif right_ctx:
            right_escaped = re.escape(right_ctx)
            pattern = entity_escaped + r'\s*' + right_escaped
        else:
            pattern = entity_escaped
        
        # Find all matches
        for match in re.finditer(pattern, text):
            # Find entity position within the match
            match_text = match.group()
            entity_pattern = r'\s*' + entity_escaped + r'\s*'
            entity_match = re.search(entity_pattern, match_text)
            
            if entity_match is None:
                continue
                        
            # Calculate actual entity bounds (excluding surrounding whitespace)
            entity_in_match = entity_match.group().strip()
            entity_pos_in_match = match_text.find(entity_in_match)
            
            entity_start = match.start() + entity_pos_in_match
            entity_end = entity_start + len(entity_in_match)
            
            return (entity_start, entity_end)
        
        return None
    
    def _is_consumed(self, span, consumed_spans):
        """Check if span overlaps with any consumed span."""
        start, end = span
        for c_start, c_end in consumed_spans:
            if start < c_end and end > c_start:
                return True
        return False