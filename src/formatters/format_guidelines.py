"""
Format guidelines for NER output formats.
"""

FORMAT_GUIDELINES = {
    'iob': (
        "Produce one line per input token containing the token followed by a single space and its IOB2 tag. "
        "The IOB2 scheme assigns B-TYPE to the first token of an entity, I-TYPE to continuation tokens of that entity, and O to tokens outside any entity. "
        "The total number of lines must equal the number of whitespace-delimited tokens in the input text. "
        "When two entities of the same type are adjacent, the second entity must begin with B- to indicate the boundary."
    ),
    
    'iob_sentence': (
        "Produce a single line of IOB2 tags separated by single spaces. "
        "Each position uses B-TYPE for entity-initial tokens, I-TYPE for continuation tokens within the same entity, and O for non-entity tokens. "
        "The count of space-separated tags must exactly equal the count of whitespace-delimited tokens in the input. "
        "Consecutive entities sharing the same type must each begin with B- to mark where one ends and another begins."
    ),
    
    'stanford_ner': (
        "Produce space-separated token-tag pairs where each pair joins the token and its label with a forward slash. "
        "Labels consist of the entity type name directly without any IOB prefixes, using O for tokens not part of any entity. "
        "Multi-token entities are represented by consecutive tokens bearing the same non-O label. "
        "Every whitespace-delimited token from the input must appear exactly once in the output sequence."
    ),
    
    'stanford_ner_iob': (
        "Produce space-separated token-tag pairs where each pair joins the token and its IOB2 label with a forward slash. "
        "Labels use B-TYPE for the first token of an entity, I-TYPE for subsequent tokens of that entity, and O for non-entity tokens. "
        "Every whitespace-delimited token from the input must appear exactly once paired with its corresponding tag. "
        "Adjacent entities of the same type require B- prefixes on both initial tokens to indicate separate entities."
    ),
    
    'csv': (
        "Produce a single line of IOB2 tags separated by commas with no surrounding whitespace. "
        "Tags follow the IOB2 scheme: B-TYPE for entity beginnings, I-TYPE for entity continuations, O for outside tokens. "
        "The number of comma-separated tags must exactly match the number of whitespace-delimited input tokens. "
        "Adjacent same-type entities each require B- on their initial token to distinguish them as separate entities."
    ),
    
    'xml_simple': (
        "Reproduce the input text with entity spans enclosed in XML tags where the tag name is the entity type. "
        "Opening and closing tags attach directly to the entity text without introducing additional whitespace. "
        "Text outside entity spans remains unchanged, preserving original spacing and punctuation. "
        "Nested entities are supported; each contiguous entity span receives its own pair of tags."
    ),
    
    'xml_ref': (
        "Reproduce the input text with entity spans enclosed in XML tags containing ref attributes that assign entity identifiers. "
        "Each tag uses the entity type as its name and includes a ref attribute with a unique identifier for that entity instance. "
        "Discontinuous entities share the same ref value across all their constituent spans. "
        "Overlapping entities at identical text positions combine their ref values as space-separated identifiers within a single tag."
    ),
    
    'json': (
        "Produce a JSON object with an entities key containing an array of entity objects. "
        "Each entity object must include label for the entity type and spans as an array of objects each with integer start and end keys for character offsets. "
        "Character offsets are zero-indexed where start is inclusive and end is exclusive. "
        "Discontinuous entities use multiple non-overlapping span objects in their spans array ordered left-to-right, each specifying one contiguous segment."
    ),

    'json_simple': (
        "Produce a JSON object where each key is an entity type and each value is a list of entity surface form strings. "
        "Only include entity types that are present in the text; omit empty labels entirely. "
        "Each unique surface form appears once per label regardless of how many times it occurs in the text. "
        "Discontinuous entities list each part as a separate string entry under the same label."
    ),

    'pubannotation': (
        "Produce a JSON object with a denotations key containing an array of annotation objects. "
        "Each denotation object requires an id key with a unique identifier, a span object with integer begin and end keys for character offsets, and an obj key for the entity type. "
        "Character offsets are zero-indexed where begin is inclusive and end is exclusive. "
        "Discontinuous entities require multiple denotation entries, each representing one contiguous segment with the same obj value."
    ),

    'pubannotation_inline': (
        "Reproduce the input text with entity spans enclosed in two consecutive bracket pairs where the first pair contains the annotated text and the second pair contains the label. "
        "Opening and closing brackets attach directly to the entity text without introducing additional whitespace. "
        "Text outside entity spans remains unchanged, preserving original spacing and punctuation. "
        "Nested entities are supported; if two consecutive bracket pairs occur naturally in non-entity text, escape the first opening bracket with a backslash to prevent misinterpretation as an annotation."
    ),

    'inline_bracketed': (
        "Reproduce the input text with entity spans enclosed in square brackets containing the entity text and its label separated by a pipe character: [Entity Text | LABEL]. "
        "Opening and closing brackets attach directly to the annotation without introducing additional whitespace beyond the spaces surrounding the pipe separator. "
        "Text outside entity spans remains unchanged, preserving original spacing and punctuation. "
        "Nested entities are supported; each contiguous entity span receives its own bracket pair."
    ),

    'gpt_ner': (
        "Reproduce the input text with entity spans enclosed by @@ prefix and ##LABEL suffix where LABEL is the entity type. "
        "Opening @@ and closing ##LABEL attach directly to the entity text without introducing additional whitespace. "
        "Text outside entity spans remains unchanged, preserving original spacing and punctuation. "
        "Nested entities are supported; each contiguous entity span receives its own @@ prefix and ##LABEL suffix."
    ),

    'symbolic_tagging': (
        "Reproduce the input text once per entity type present, with each reproduction on its own line prefixed by the entity type and a colon. "
        "On each line, insert [ before and ] after every entity span of that type only; all other text remains unchanged. "
        "Nested entities of the same type are represented with nested brackets; each contiguous entity span receives its own pair of brackets. "
        "If the input text contains newlines, replace them with spaces to preserve the one-line-per-type structure."
    ),
    
    'csv_complex': (
        "Produce one line per entity with three comma-separated fields: a unique entity identifier, the entity type, and character offset spans. "
        "Spans are formatted as start:end pairs using zero-indexed character offsets, with multiple spans for discontinuous entities joined by semicolons. "
        "The start offset is inclusive and the end offset is exclusive. "
        "When no entities exist in the input, output: NONE"
    ),
    
    'brat': (
        "Produce one line per entity in standoff annotation format with two tab-separated components: entity identifier and type with offsets. "
        "The second component contains the entity type followed by space-separated zero-indexed character offsets as start-end pairs. "
        "Discontinuous entities list multiple start-end pairs separated by semicolons within the offset specification. "
        "Entity identifiers follow the pattern T followed by a sequential integer starting from one."
        "When no entities exist in the input, output: NONE"
    ),
    
    'slender': (
        "Produce a sequence of entity annotations where each annotation contains the entity surface text, a pipe character surrounded by spaces, and the entity type label. "
        "Multiple entity annotations are separated by semicolons with optional surrounding spaces. "
        "This format contains no positional information; entities are identified solely by their surface text and type. "
        "When no entities exist in the input, output: NONE"
    ),
    
    'cane_base': (
        "Produce a sequence of entity annotations starting with 'Entities:' on the first line, followed by entity annotations where each annotation contains the entity surface text enclosed in backticks, the phrase 'is a' (or 'is an'), and the entity type label. "
        "Multiple entity annotations are separated by newlines. "
        "This format contains no positional information; entities are identified solely by their surface text and type. "
        "When no entities exist in the input, output: NONE"
    ),

    # Variation of cane_base with discontinuous entity support
    'cane_disco': (
        "Produce output starting with 'Entities:' on the first line, followed by one line per entity as a natural language statement: `entity_text` is a LABEL. "
        "For discontinuous entities, list all parts ordered left-to-right joined by 'and': `part1` and `part2` is a LABEL. "
        "This format contains no positional information; entities are identified solely by their surface text and type. "
        "Multiple entity annotations are separated by newlines and each occupies exactly one line. "
        "When no entities exist in the input, output: NONE"
    ),

    # CANE-Full: verbatim context per entity part, supports discontinuous entities
    'cane_full': (
        "Produce output starting with 'Entities:' on the first line, followed by one line per entity as a natural language statement: `entity_text` is a LABEL found in `left entity right`. "
        "The entity text appears verbatim within the backtick-delimited context string, surrounded by its actual left and right context. "
        "For discontinuous entities, list all non-overlapping parts ordered left-to-right with 'and': `part1` and `part2` is a LABEL found in `left1 part1 right1` and `left2 part2 right2`. "
        "Each context string contains the exact text as it appears in the document for that specific part, including the entity part itself. "
        "When no entities exist in the input, output: NONE"
    ),

    # CANE-Context: CANE-Full without discontinuous entities
    'cane_context': (
        "Produce output starting with 'Entities:' on the first line, followed by one line per entity as a natural language statement: `entity_text` is a LABEL found in `left entity right`. "
        "The entity text appears verbatim within the backtick-delimited context string, surrounded by its actual left and right context. "
        "The context string contains the exact text as it appears in the document, including the entity itself and its immediately adjacent tokens. "
        "Each entity occupies exactly one line with a single backtick-delimited entity and a single backtick-delimited context string. "
        "When no entities exist in the input, output: NONE"
    ),
    
    # ---- Ablation ----
    'cane_base_abl_newline': (
        "Produce a sequence of entity annotations starting with 'Entities:' on the first line, followed by entity annotations where each annotation contains the entity surface text enclosed in backticks, the phrase 'is a' (or 'is an'), and the entity type label. "
        "Multiple entity annotations are separated by semicolons. "
        "This format contains no positional information; entities are identified solely by their surface text and type. "
        "When no entities exist in the input, output: NONE"
    ),

    'cane_base_abl_backticks': (
        "Produce a sequence of entity annotations starting with 'Entities:' on the first line, followed by entity annotations where each annotation contains the entity surface text enclosed in double quotes, the phrase 'is a' (or 'is an'), and the entity type label. "
        "Multiple entity annotations are separated by newlines. "
        "This format contains no positional information; entities are identified solely by their surface text and type. "
        "When no entities exist in the input, output: NONE"
    ),

    'cane_base_abl_header': (
        "Produce a sequence of entity annotations where each annotation contains the entity surface text enclosed in backticks, the phrase 'is a' (or 'is an'), and the entity type label. "
        "Multiple entity annotations are separated by newlines. "
        "This format contains no positional information; entities are identified solely by their surface text and type. "
        "When no entities exist in the input, output: NONE"
    ),

    # PT format (He and Choi, 2023)
    'pt_format': (
        "Produce a sequence of entity annotations where each annotation contains the entity surface text enclosed in double quotes, the phrase 'is a' (or 'is an'), and the entity type label. "
        "Multiple entity annotations are separated by semicolons. "
        "This format contains no positional information; entities are identified solely by their surface text and type. "
        "When no entities exist in the input, output: NONE"
    ),
}