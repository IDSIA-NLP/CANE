from parsers.ner.csv_parser import CSVParser
from parsers.ner.iob_parser import IOBParser
from parsers.ner.iob_sentence_parser import IOBSentenceParser
from parsers.ner.pubannotation_parser import PubAnnotationParser
from parsers.ner.pubannotation_inline_parser import PubAnnotationInlineParser
from parsers.ner.inline_bracketed_parser import InlineBracketedParser
from parsers.ner.stanford_ner_iob_parser import StanfordNERIOBParser
from parsers.ner.stanford_ner_parser import StanfordNERParser

from parsers.ner.xml_simple_parser import XMLSimpleParser
from parsers.ner.xml_ref_parser import XMLRefParser

from parsers.ner.brat_parser import BRATParser
from parsers.ner.csv_complex_parser import CSVComplexParser
from parsers.ner.json_parser import JSONParser
from parsers.ner.json_simple_parser import JSONSimpleParser

from parsers.ner.gpt_ner_parser import GPTNERParser
from parsers.ner.symbolic_tagging_parser import SymbolicTaggingParser

from parsers.ner.slender_parser import SLENDERParser

from parsers.ner.cane_base_parser import CANEBaseParser
from parsers.ner.cane_full_parser import CANEFullParser
from parsers.ner.cane_context_parser import CANEContextParser
from parsers.ner.cane_disco_parser import CANEDiscoParser

# Ablation format parsers
from parsers.ner.cane_base_abl_backticks_parser import CANEBaseAblBackticksParser
from parsers.ner.cane_base_abl_header_parser import CANEBaseAblHeaderParser
from parsers.ner.cane_base_abl_newline_parser import CANEBaseAblNewlineParser
from parsers.ner.pt_format_parser import PTFormatParser


class ParserFactory:
    @staticmethod
    def get_parser(task: str, output_format: str):
        if task == 'ner':
            if output_format == 'brat':
                return BRATParser()
            elif output_format == 'csv':
                return CSVParser()
            elif output_format == 'iob':
                return IOBParser()
            elif output_format == 'iob_sentence':
                return IOBSentenceParser()
            elif output_format == 'pubannotation':
                return PubAnnotationParser()
            elif output_format == 'pubannotation_inline':
                return PubAnnotationInlineParser()
            elif output_format == 'inline_bracketed':
                return InlineBracketedParser()
            elif output_format == 'gpt_ner':
                return GPTNERParser()
            elif output_format == 'symbolic_tagging':
                return SymbolicTaggingParser()
            elif output_format == 'stanford_ner_iob':
                return StanfordNERIOBParser()
            elif output_format == 'stanford_ner':
                return StanfordNERParser()
            elif output_format == 'xml_simple':
                return XMLSimpleParser()
            elif output_format == 'xml_ref':
                return XMLRefParser()
            elif output_format == 'csv_complex':
                return CSVComplexParser()
            elif output_format == 'json':
                return JSONParser()
            elif output_format == 'json_simple':
                return JSONSimpleParser()
            elif output_format == 'slender':
                return SLENDERParser()
            elif output_format == 'cane_base':
                return CANEBaseParser()
            elif output_format == 'cane_disco':
                return CANEDiscoParser()
            # CANE context-anchored formats
            elif output_format == 'cane_full':
                return CANEFullParser()
            elif output_format == 'cane_context':
                return CANEContextParser()
            # Ablation format parsers
            elif output_format == 'cane_base_abl_backticks':
                return CANEBaseAblBackticksParser()
            elif output_format == 'cane_base_abl_header':
                return CANEBaseAblHeaderParser()
            elif output_format == 'cane_base_abl_newline':
                return CANEBaseAblNewlineParser()
            elif output_format == 'pt_format':
                return PTFormatParser()
            else:
                raise ValueError(f"Unsupported output format for NER: {output_format}")

        else:
            raise ValueError(f"Unsupported task type: {task}")