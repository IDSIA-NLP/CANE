from formatters.input_formatters.ner.whitespace_input_formatter import WhitespaceInputFormatter

from formatters.output_formatters.ner.iob_output_formatter import IOBOutputFormatter
from formatters.output_formatters.ner.iob_sentence_output_formatter import IOBSentenceOutputFormatter
from formatters.output_formatters.ner.stanford_ner_output_formatter import StanfordNerOutputFormatter
from formatters.output_formatters.ner.stanford_ner_iob_output_formatter import StanfordNerIOBOutputFormatter
from formatters.output_formatters.ner.csv_output_formatter import CSVOutputFormatter
from formatters.output_formatters.ner.xml_simple_output_formatter import XMLSimpleOutputFormatter
from formatters.output_formatters.ner.xml_ref_output_formatter import XMLRefOutputFormatter
from formatters.output_formatters.ner.pubannotation_output_formatter import PubAnnotationOutputFormatter
from formatters.output_formatters.ner.pubannotation_inline_output_formatter import PubAnnotationInlineOutputFormatter
from formatters.output_formatters.ner.inline_bracketed_output_formatter import InlineBracketedOutputFormatter

from formatters.output_formatters.ner.brat_output_formatter import BRATOutputFormatter
from formatters.output_formatters.ner.csv_complex_output_formatter import CSVComplexOutputFormatter
from formatters.output_formatters.ner.json_output_formatter import JSONOutputFormatter
from formatters.output_formatters.ner.json_simple_output_formatter import JSONSimpleOutputFormatter

from formatters.output_formatters.ner.gpt_ner_output_formatter import GPTNEROutputFormatter
from formatters.output_formatters.ner.symbolic_tagging_output_formatter import SymbolicTaggingOutputFormatter

from formatters.output_formatters.ner.slender_output_formatter import SLENDEROutputFormatter

from formatters.output_formatters.ner.cane_base_output_formatter import CANEBaseOutputFormatter
from formatters.output_formatters.ner.cane_full_output_formatter import CANEFullOutputFormatter
from formatters.output_formatters.ner.cane_context_output_formatter import CANEContextOutputFormatter
from formatters.output_formatters.ner.cane_disco_output_formatter import CANEDiscoOutputFormatter

# Ablation formatters
from formatters.output_formatters.ner.cane_base_abl_backticks_output_formatter import CANEBaseAblBackticksOutputFormatter
from formatters.output_formatters.ner.cane_base_abl_header_output_formatter import CANEBaseAblHeaderOutputFormatter
from formatters.output_formatters.ner.cane_base_abl_newline_output_formatter import CANEBaseAblNewlineOutputFormatter
from formatters.output_formatters.ner.pt_format_output_formatter import PTFormatOutputFormatter

from loguru import logger


class FormatterFactory:
    @staticmethod
    def get_formatter(format_type, task, input_or_output):
        if input_or_output == 'input':
            if task == 'ner':
                if format_type == 'white_space':
                    return WhitespaceInputFormatter()

            else:
                logger.error(f"Unsupported input format for task. Format: {format_type}, Task: {task}")
                raise ValueError(f"Unsupported input format for task. Format: {format_type}, Task: {task}")

        elif input_or_output == 'output':
            if task == 'ner':
                if format_type == 'brat':
                    return BRATOutputFormatter()
                elif format_type == 'iob':
                    return IOBOutputFormatter()
                elif format_type == 'iob_sentence':
                    return IOBSentenceOutputFormatter()
                elif format_type == 'stanford_ner':
                    return StanfordNerOutputFormatter()
                elif format_type == 'stanford_ner_iob':
                    return StanfordNerIOBOutputFormatter()
                elif format_type == 'csv':
                    return CSVOutputFormatter()
                elif format_type == 'xml_simple':
                    return XMLSimpleOutputFormatter()
                elif format_type == 'xml_ref':
                    return XMLRefOutputFormatter()
                elif format_type == 'pubannotation':
                    return PubAnnotationOutputFormatter()
                elif format_type == 'pubannotation_inline':
                    return PubAnnotationInlineOutputFormatter()
                elif format_type == 'inline_bracketed':
                    return InlineBracketedOutputFormatter()
                elif format_type == 'gpt_ner':
                    return GPTNEROutputFormatter()
                elif format_type == 'symbolic_tagging':
                    return SymbolicTaggingOutputFormatter()
                elif format_type == 'csv_complex':
                    return CSVComplexOutputFormatter()
                elif format_type == 'json':
                    return JSONOutputFormatter()
                elif format_type == 'json_simple':
                    return JSONSimpleOutputFormatter()
                elif format_type == 'slender':
                    return SLENDEROutputFormatter()
                elif format_type == 'cane_base':
                    return CANEBaseOutputFormatter()
                elif format_type == 'cane_disco':
                    return CANEDiscoOutputFormatter()
                # CANE context-anchored formats
                elif format_type == 'cane_full':
                    return CANEFullOutputFormatter()
                elif format_type == 'cane_context':
                    return CANEContextOutputFormatter()
                # Ablation formatters
                elif format_type == 'cane_base_abl_backticks':
                    return CANEBaseAblBackticksOutputFormatter()
                elif format_type == 'cane_base_abl_header':
                    return CANEBaseAblHeaderOutputFormatter()
                elif format_type == 'cane_base_abl_newline':
                    return CANEBaseAblNewlineOutputFormatter()
                elif format_type == 'pt_format':
                    return PTFormatOutputFormatter()

            else:
                logger.error(f"Unsupported output format for task. Format: {format_type}, Task: {task}")
                raise ValueError(f"Unsupported output format for task. Format: {format_type}, Task: {task}")

        raise ValueError(f"Unsupported format or task. Format: {format_type}, Task: {task}")