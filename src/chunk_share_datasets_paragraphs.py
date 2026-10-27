"""
Split prepared BRAT datasets (share2013, share2014) into one document per
paragraph, written to <name>_paragraphs.

Paragraphs are separated by blank lines. Annotations crossing a paragraph
boundary are dropped, offsets are made relative to the paragraph, and spans
are checked against the text before writing.

Usage:
    python chunk_share_datasets_paragraphs.py
    python chunk_share_datasets_paragraphs.py --input-dir ../data/datasets/prepared --datasets share2013 share2014
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple
from loguru import logger


def find_paragraph_boundaries(text: str) -> List[Tuple[int, int]]:
    """
    Return (start, end) offsets of each non-empty paragraph in the original text.

    Text should be read with newline='' so \r\n is kept and offsets match the .ann spans.
    """
    # One or more blank lines, \n or \r\n
    paragraph_split = re.split(r'(\r?\n\r?\n)+', text)
    
    boundaries = []
    current_offset = 0
    
    for segment in paragraph_split:
        segment_end = current_offset + len(segment)
        
        # Check if this segment is a separator (only whitespace/newlines)
        if segment.strip():
            # Non-empty paragraph: record its boundaries in the original text
            boundaries.append((current_offset, segment_end))
        
        current_offset = segment_end
    
    return boundaries


def load_annotations(ann_file: Path) -> List[Dict]:
    """Load T-lines from a .ann file as dicts with ann_id, label, spans, text."""
    annotations = []
    
    if not ann_file.exists():
        return annotations
    
    with open(ann_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('T'):
                continue
            
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            
            ann_id, type_span, entity_text = parts
            type_parts = type_span.split(' ', 1)
            
            if len(type_parts) != 2:
                continue
            
            label = type_parts[0]
            span_str = type_parts[1]
            
            # Parse spans - handle discontinuous with semicolon
            spans = []
            for span_part in span_str.split(';'):
                offsets = span_part.strip().split()
                if len(offsets) == 2:
                    try:
                        start, end = int(offsets[0]), int(offsets[1])
                        spans.append((start, end))
                    except ValueError:
                        logger.warning(f"Invalid span offsets in {ann_file.name}: {span_part}")
                        continue
            
            if spans:
                annotations.append({
                    'ann_id': ann_id,
                    'label': label,
                    'spans': spans,
                    'text': entity_text
                })
    
    return annotations


def annotation_within_paragraph(annotation: Dict, para_start: int, para_end: int) -> bool:
    """True if every span of the annotation (including discontinuous fragments) is inside the paragraph."""
    for span_start, span_end in annotation['spans']:
        if span_start < para_start or span_end > para_end:
            return False
    return True


def reset_annotation_offsets(annotation: Dict, para_start: int) -> Dict:
    """Return a copy of the annotation with offsets relative to the paragraph start."""
    adjusted_spans = [
        (start - para_start, end - para_start)
        for start, end in annotation['spans']
    ]
    return {
        'ann_id': annotation['ann_id'],
        'label': annotation['label'],
        'spans': adjusted_spans,
        'text': annotation['text']
    }


def validate_annotation(annotation: Dict, paragraph_text: str) -> bool:
    """
    Check that the spans extract the stored annotation text.

    Newlines are replaced with spaces before comparing, since the stored text
    was cleaned with .replace('\\n', ' ').strip().
    """
    extracted_parts = []
    for start, end in annotation['spans']:
        if start < 0 or end > len(paragraph_text) or start >= end:
            return False
        extracted_parts.append(
            paragraph_text[start:end].replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ').strip()
        )
    
    extracted = ' '.join(extracted_parts)
    stored = annotation['text'].strip()
    
    return extracted == stored


def write_brat_document(output_dir: Path, doc_id: str, text: str, annotations: List[Dict]):
    """Write {doc_id}.txt and {doc_id}.ann to output_dir."""
    # Write text file - preserve original line endings
    txt_path = output_dir / f"{doc_id}.txt"
    with open(txt_path, 'w', encoding='utf-8', newline='') as f:
        f.write(text)
    
    # Write annotation file
    ann_path = output_dir / f"{doc_id}.ann"
    with open(ann_path, 'w', encoding='utf-8') as f:
        for idx, ann in enumerate(annotations, start=1):
            spans = ann['spans']
            label = ann['label']
            entity_text = ann['text']
            
            # Format spans - discontinuous uses semicolon
            if len(spans) == 1:
                span_str = f"{spans[0][0]} {spans[0][1]}"
            else:
                span_str = ";".join(f"{s[0]} {s[1]}" for s in spans)
            
            f.write(f"T{idx}\t{label} {span_str}\t{entity_text}\n")


def process_brat_directory(input_dir: Path, output_dir: Path):
    """Split every document in input_dir by paragraph and write them as {doc_id}_para{N}."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    txt_files = sorted(input_dir.glob("*.txt"))
    
    if not txt_files:
        logger.warning(f"No .txt files found in {input_dir}")
        return
    
    logger.info(f"Processing {len(txt_files)} documents from {input_dir}")
    
    total_paragraphs_written = 0
    total_annotations_kept = 0
    total_annotations_dropped_boundary = 0
    total_annotations_dropped_validation = 0
    total_paragraphs_skipped_empty = 0
    
    for txt_file in txt_files:
        ann_file = txt_file.with_suffix('.ann')
        doc_id = txt_file.stem
        
        # Read text preserving original line endings so offsets stay correct
        with open(txt_file, 'r', encoding='utf-8', newline='') as f:
            text = f.read()
        
        # Load annotations
        annotations = load_annotations(ann_file)
        
        # Find paragraph boundaries
        paragraphs = find_paragraph_boundaries(text)
        
        if not paragraphs:
            logger.warning(f"No paragraphs found in {txt_file.name}, skipping")
            continue
        
        doc_paragraphs_written = 0
        
        for para_idx, (para_start, para_end) in enumerate(paragraphs):
            paragraph_text = text[para_start:para_end]
            
            # Skip paragraphs that are only whitespace
            if not paragraph_text.strip():
                total_paragraphs_skipped_empty += 1
                continue
            
            # Filter annotations: keep only those fully within this paragraph
            para_annotations = []
            for ann in annotations:
                if annotation_within_paragraph(ann, para_start, para_end):
                    para_annotations.append(ann)
                else:
                    # Count only annotations crossing the boundary, not those fully outside
                    any_span_overlaps = any(
                        span_start < para_end and span_end > para_start
                        for span_start, span_end in ann['spans']
                    )
                    if any_span_overlaps:
                        total_annotations_dropped_boundary += 1
                        logger.debug(
                            f"Dropped boundary-crossing annotation in {doc_id} para {para_idx}: "
                            f"{ann['text']!r} spans {ann['spans']}"
                        )
            
            # Reset offsets to be relative to paragraph start
            para_annotations_reset = [
                reset_annotation_offsets(ann, para_start)
                for ann in para_annotations
            ]
            
            # Validate all annotations before writing
            valid_annotations = []
            for ann in para_annotations_reset:
                if validate_annotation(ann, paragraph_text):
                    valid_annotations.append(ann)
                    total_annotations_kept += 1
                else:
                    total_annotations_dropped_validation += 1
                    logger.warning(
                        f"Validation failed for annotation in {doc_id} para {para_idx}: "
                        f"label={ann['label']!r}, stored={ann['text']!r}, "
                        f"spans={ann['spans']}"
                    )
            
            # Write paragraph as new BRAT document
            para_doc_id = f"{doc_id}_para{para_idx:03d}"
            write_brat_document(output_dir, para_doc_id, paragraph_text, valid_annotations)
            
            doc_paragraphs_written += 1
            total_paragraphs_written += 1
        
        logger.debug(f"{doc_id}: {len(paragraphs)} paragraphs, {doc_paragraphs_written} written")
    
    logger.info(f"Done processing {input_dir.name}")
    logger.info(f"  Paragraphs written       : {total_paragraphs_written}")
    logger.info(f"  Paragraphs skipped (empty): {total_paragraphs_skipped_empty}")
    logger.info(f"  Annotations kept         : {total_annotations_kept}")
    logger.info(f"  Annotations dropped (boundary crossing): {total_annotations_dropped_boundary}")
    logger.info(f"  Annotations dropped (validation failed): {total_annotations_dropped_validation}")


def main():
    parser = argparse.ArgumentParser(
        description="Chunk BRAT datasets into paragraph-level sub-documents"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="../data/datasets/prepared",
        help="Base directory containing prepared BRAT datasets"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["share2013", "share2014"],
        help="Dataset names to chunk (must exist as subdirectories of input-dir)"
    )
    
    args = parser.parse_args()
    input_base = Path(args.input_dir)
    
    for dataset_name in args.datasets:
        dataset_input_dir = input_base / dataset_name
        
        if not dataset_input_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_input_dir}")
            continue
        
        # Output directory name reflects the chunking strategy
        output_dataset_name = f"{dataset_name}_paragraphs"
        
        logger.info(f"Chunking {dataset_name} into {output_dataset_name}")
        
        # Handle datasets with train/test subdirectories
        has_splits = any((dataset_input_dir / split).exists() for split in ["train", "test"])
        
        if has_splits:
            for split in ["train", "test"]:
                split_input = dataset_input_dir / split
                if not split_input.exists():
                    continue
                split_output = input_base / output_dataset_name / split
                logger.info(f"  Processing split: {split}")
                process_brat_directory(split_input, split_output)
        else:
            # Flat directory
            output_dir = input_base / output_dataset_name
            process_brat_directory(dataset_input_dir, output_dir)
        
        logger.info(f"Done: {dataset_name} -> {output_dataset_name}")


if __name__ == "__main__":
    main()