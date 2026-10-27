"""
Convert the original CADEC, SHARE and GENIA datasets to BRAT standoff format.

Usage:
    python prepare_datasets.py --dataset cadec
    python prepare_datasets.py --dataset share2013
    python prepare_datasets.py --dataset share2014
    python prepare_datasets.py --dataset genia
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict
from loguru import logger


class BRATWriter:
    """Handles writing BRAT standoff format files."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_document(self, doc_id: str, text: str, annotations: List[Dict]):
        """Write {doc_id}.txt and {doc_id}.ann. Annotations are dicts with spans, label and text."""
        # Write text file
        # newline='' keeps \r\n so BRAT offsets stay correct
        txt_path = self.output_dir / f"{doc_id}.txt"
        with open(txt_path, 'w', encoding='utf-8', newline='') as f:
            f.write(text)
        
        # Write annotation file
        ann_path = self.output_dir / f"{doc_id}.ann"
        with open(ann_path, 'w', encoding='utf-8') as f:
            for idx, ann in enumerate(annotations, start=1):
                spans = ann['spans']
                label = ann['label']
                text_span = ann['text']
                
                # Format spans for BRAT (discontinuous uses semicolon)
                if len(spans) == 1:
                    span_str = f"{spans[0][0]} {spans[0][1]}"
                else:
                    span_str = ";".join(f"{s[0]} {s[1]}" for s in spans)
                
                # Write annotation line: T1\tLabel start end\ttext
                f.write(f"T{idx}\t{label} {span_str}\t{text_span}\n")
        
        logger.debug(f"Wrote {doc_id}: {len(annotations)} annotations")


class CADECConverter:
    """Convert CADEC dataset from original BRAT format to cleaned BRAT format."""
    
    def __init__(self, input_dir: Path, output_dir: Path, entity_types: List[str] = None):
        self.input_ann_dir = input_dir / "original"
        self.input_text_dir = input_dir / "text"
        self.output_dir = output_dir
        self.entity_types = entity_types or ["ADR"]  # Default to ADR
        
        if not self.input_ann_dir.exists():
            raise ValueError(f"Annotation directory not found: {self.input_ann_dir}")
        if not self.input_text_dir.exists():
            raise ValueError(f"Text directory not found: {self.input_text_dir}")
    
    def extract_indices_from_brat(self, span_str: str) -> List[int]:
        """Extract numeric indices from BRAT span string (handles discontinuous)."""
        import re
        indices = re.findall(r"\d+", span_str)
        return sorted([int(i) for i in indices])
    
    def convert(self):
        """Convert CADEC annotations to cleaned BRAT format."""
        writer = BRATWriter(self.output_dir)
        ann_files = list(self.input_ann_dir.glob("*.ann"))
        
        logger.info(f"Converting {len(ann_files)} CADEC documents...")
        
        for ann_file in ann_files:
            doc_id = ann_file.stem
            txt_file = self.input_text_dir / f"{doc_id}.txt"
            
            if not txt_file.exists():
                logger.warning(f"Text file not found for {doc_id}, skipping")
                continue
            
            # Read text
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Read annotations
            annotations = []
            with open(ann_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line[0] != 'T':
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) != 3:
                        continue
                    
                    # Parse: T1\tLabel start end;start end\ttext
                    ann_id, type_span, mention_text = parts
                    type_parts = type_span.split(' ', 1)
                    label = type_parts[0]
                    
                    # Filter by entity type
                    if self.entity_types and label not in self.entity_types:
                        continue
                    
                    # Extract indices
                    span_str = type_parts[1] if len(type_parts) > 1 else ""
                    indices = self.extract_indices_from_brat(span_str)
                    
                    if len(indices) < 2:
                        logger.warning(f"Invalid span in {doc_id}: {line}")
                        continue
                    
                    # Convert to list of (start, end) tuples
                    spans = [(indices[i], indices[i+1]) for i in range(0, len(indices), 2)]
                    
                    # Validate text matches
                    extracted_text = " ".join(
                        text[s[0]:s[1]].replace("\n", " ").strip() 
                        for s in spans
                    )
                    
                    annotations.append({
                        'spans': spans,
                        'label': label,
                        'text': extracted_text
                    })
            
            writer.write_document(doc_id, text, annotations)
        
        logger.info(f"Converted {len(ann_files)} CADEC documents to {self.output_dir}")


class SHARE2013Converter:
    """Convert SHARE 2013 dataset to BRAT format."""
    
    def __init__(self, input_dir: Path, output_dir: Path):
        self.train_ann_dir = input_dir / "train" / "ann"
        self.train_text_dir = input_dir / "train" / "text"
        self.test_ann_dir = input_dir / "test" / "ann"
        self.test_text_dir = input_dir / "test" / "text"
        self.output_dir = output_dir
    
    def convert_train(self):
        """Convert training set from Knowtator XML format."""
        logger.info("Converting SHARE 2013 training set...")
        writer = BRATWriter(self.output_dir / "train")
        
        xml_files = list(self.train_ann_dir.glob("*.knowtator.xml"))
        
        for xml_file in xml_files:
            # Strip '.txt.knowtator.xml' so doc_id does not retain the '.txt' suffix
            doc_id = xml_file.name.replace(".txt.knowtator.xml", "")
            txt_file = self.train_text_dir / (doc_id + ".txt")
            
            if not txt_file.exists():
                logger.warning(f"Text file not found for {doc_id}, skipping")
                continue
            
            # Read text
            # newline='' keeps \r\n so character offsets match the annotations
            with open(txt_file, 'r', encoding='utf-8', newline='') as f:
                text = f.read()
            
            # Parse XML
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
            except Exception as e:
                logger.error(f"Failed to parse {xml_file}: {e}")
                continue
            
            annotations = []
            for mention in root.findall("annotation"):
                # Extract spans
                indices = []
                for span in mention.findall("span"):
                    indices.append(int(span.get("start")))
                    indices.append(int(span.get("end")))
                
                if not indices:
                    continue
                
                indices = sorted(indices)
                spans = [(indices[i], indices[i+1]) for i in range(0, len(indices), 2)]
                
                # Extract text from spans
                mention_texts = []
                gap_texts = []
                for i in range(0, len(indices), 2):
                    mention_texts.append(
                        text[indices[i]:indices[i+1]].replace("\n", " ").strip()
                    )
                
                # Check if discontinuous (has gaps with no text)
                for i in range(1, len(indices) - 2, 2):
                    gap_texts.append(
                        text[indices[i]:indices[i+1]].replace("\n", " ").strip()
                    )
                
                # If gaps are empty, merge to continuous span
                if len(indices) > 2 and all(not g for g in gap_texts):
                    logger.debug(f"Merging pseudo-discontinuous span in {doc_id}")
                    spans = [(indices[0], indices[-1])]
                
                mention_text = " ".join(mention_texts)
                
                annotations.append({
                    'spans': spans,
                    'label': 'Disorder',
                    'text': mention_text
                })
            
            writer.write_document(doc_id, text, annotations)
        
        logger.info(f"Converted {len(xml_files)} training documents")
    
    def convert_test(self):
        """Convert test set from pipe-delimited format."""
        logger.info("Converting SHARE 2013 test set...")
        writer = BRATWriter(self.output_dir / "test")
        
        # Read all annotations
        mentions_by_doc = {}
        for ann_file in self.test_ann_dir.iterdir():
            if not ann_file.is_file():
                continue
            
            
            txt_file = self.test_text_dir / ann_file.name
            doc_id = ann_file.name.split(".")[0]
            
            if not txt_file.exists():
                continue
            
            # newline='' keeps \r\n so character offsets match the annotations
            with open(txt_file, 'r', encoding='utf-8', newline='') as f:
                text = f.read()
            
            annotations = []
            with open(ann_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split("||")
                    if len(parts) < 4:
                        continue
                    
                    if parts[1] != "Disease_Disorder":
                        continue
                    
                    # Extract indices (all after index 3)
                    indices = sorted([int(i) for i in parts[3:]])
                    
                    if len(indices) < 2:
                        continue
                    
                    spans = [(indices[i], indices[i+1]) for i in range(0, len(indices), 2)]
                    
                    # Extract text
                    mention_texts = []
                    gap_texts = []
                    for i in range(0, len(indices), 2):
                        mention_texts.append(
                            text[indices[i]:indices[i+1]].replace("\n", " ").strip()
                        )
                    
                    for i in range(1, len(indices) - 2, 2):
                        gap_texts.append(
                            text[indices[i]:indices[i+1]].replace("\n", " ").strip()
                        )
                    
                    # Merge if no real gaps
                    if len(indices) > 2 and all(not g for g in gap_texts):
                        spans = [(indices[0], indices[-1])]
                    
                    mention_text = " ".join(mention_texts)
                    
                    annotations.append({
                        'spans': spans,
                        'label': 'Disorder',
                        'text': mention_text
                    })
            
            writer.write_document(doc_id, text, annotations)
        
        logger.info(f"Converted test documents")
    
    def convert(self):
        """Convert the test set."""
        # self.convert_train()
        self.convert_test()


class SHARE2014Converter:
    """Convert SHARE 2014 dataset to BRAT format."""
    
    def __init__(self, input_dir: Path, output_dir: Path):
        self.train_ann_dir = input_dir / "train" / "ann"
        self.train_text_dir = input_dir / "train" / "text"
        self.test_ann_dir = input_dir / "test" / "ann"
        self.test_text_dir = input_dir / "test" / "text"
        self.output_dir = output_dir
    
    def convert_split(self, ann_dir: Path, text_dir: Path, split_name: str):
        """Convert a dataset split from pipe-delimited format."""
        logger.info(f"Converting SHARE 2014 {split_name} set...")
        writer = BRATWriter(self.output_dir / split_name)
        
        for ann_file in ann_dir.glob("*.pipe*"):
            
            txt_file = text_dir / ann_file.name.replace(".pipe", "")
            doc_id = ann_file.name.split(".")[0]
            
            if not txt_file.exists():
                logger.warning(f"Text file not found for {doc_id}, skipping")
                continue
            
            # Read text
            # newline='' keeps \r\n so character offsets match the annotations
            with open(txt_file, 'r', encoding='utf-8', newline='') as f:
                text = f.read()
            
            annotations = []
            with open(ann_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split("|")
                    if len(parts) < 2:
                        continue
                    
                    # Extract indices from span string
                    import re
                    indices = re.findall(r"\d+", parts[1])
                    indices = sorted([int(i) for i in indices])
                    
                    if len(indices) < 2:
                        continue
                    
                    spans = [(indices[i], indices[i+1]) for i in range(0, len(indices), 2)]
                    
                    # Extract text
                    mention_texts = []
                    gap_texts = []
                    for i in range(0, len(indices), 2):
                        mention_texts.append(
                            text[indices[i]:indices[i+1]].replace("\n", " ").strip()
                        )
                    
                    for i in range(1, len(indices) - 2, 2):
                        gap_texts.append(
                            text[indices[i]:indices[i+1]].replace("\n", " ").strip()
                        )
                    
                    # Merge if no real gaps
                    if len(indices) > 2 and all(not g for g in gap_texts):
                        spans = [(indices[0], indices[-1])]
                    
                    mention_text = " ".join(mention_texts)
                    
                    annotations.append({
                        'spans': spans,
                        'label': 'Disorder',
                        'text': mention_text
                    })
            
            writer.write_document(doc_id, text, annotations)
        
        logger.info(f"Converted {split_name} documents")
    
    def convert(self):
        """Convert the test set."""
        # self.convert_split(self.train_ann_dir, self.train_text_dir, "train")
        self.convert_split(self.test_ann_dir, self.test_text_dir, "test")


class GENIAConverter:
    """Convert GENIA corpus from inline XML format to BRAT standoff format."""
    
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
    
    def extract_sem_label(self, sem_attr: str) -> str:
        """
        Get a plain label from a GENIA sem attribute, e.g.
        G#protein_molecule or (AND G#protein_molecule G#protein_molecule) -> protein_molecule
        """
        if not sem_attr:
            return "entity"
        
        # Handle predicate-argument structures
        if sem_attr.startswith("("):
            # Extract first concept from complex expression
            import re
            match = re.search(r"G#(\w+)", sem_attr)
            if match:
                return match.group(1)
            return "entity"
        
        # Simple case: G#label
        if sem_attr.startswith("G#"):
            return sem_attr[2:]  # Remove G# prefix
        
        return sem_attr
    
    def extract_text_and_annotations(self, element, current_offset: int = 0):
        """Recursively collect text and <cons> annotations. Returns (text, annotations, end_offset)."""
        text_parts = []
        annotations = []
        offset = current_offset
        
        # Add text before first child
        if element.text:
            text_parts.append(element.text)
            offset += len(element.text)
        
        # Process children
        for child in element:
            if child.tag == "cons":
                # Record start of this entity
                entity_start = offset
                
                # Recursively process nested cons
                child_text, child_annotations, offset = self.extract_text_and_annotations(child, offset)
                text_parts.append(child_text)
                
                # Record this entity
                entity_end = offset
                sem_attr = child.get("sem", "")
                label = self.extract_sem_label(sem_attr)
                
                annotations.append({
                    'spans': [(entity_start, entity_end)],
                    'label': label,
                    'text': child_text
                })
                
                # Add nested annotations
                annotations.extend(child_annotations)
            else:
                # Process other elements recursively
                child_text, child_annotations, offset = self.extract_text_and_annotations(child, offset)
                text_parts.append(child_text)
                annotations.extend(child_annotations)
            
            # Add tail text after child
            if child.tail:
                text_parts.append(child.tail)
                offset += len(child.tail)
        
        full_text = "".join(text_parts)
        return full_text, annotations, offset
    
    def convert_article(self, article_element):
        """Convert one GENIA article (title and abstract) to (doc_id, text, annotations)."""
        # Extract MEDLINE ID
        bibliomisc = article_element.find(".//bibliomisc")
        doc_id = "unknown"
        if bibliomisc is not None and bibliomisc.text:
            # Format: MEDLINE:12345678
            doc_id = bibliomisc.text.strip().replace(":", "_").replace("/", "_")
        
        # Extract title and abstract
        text_parts = []
        all_annotations = []
        current_offset = 0
        
        # Process title
        title = article_element.find("title")
        if title is not None:
            title_text, title_anns, current_offset = self.extract_text_and_annotations(title, current_offset)
            text_parts.append(title_text)
            all_annotations.extend(title_anns)
            text_parts.append("\n\n")  # Add spacing
            current_offset += 2
        
        # Process abstract sentences
        abstract = article_element.find("abstract")
        if abstract is not None:
            sentences = abstract.findall("sentence")
            for i, sentence in enumerate(sentences):
                sent_text, sent_anns, current_offset = self.extract_text_and_annotations(sentence, current_offset)
                text_parts.append(sent_text)
                all_annotations.extend(sent_anns)
                
                # Add space between sentences
                if i < len(sentences) - 1:
                    text_parts.append(" ")
                    current_offset += 1
        
        full_text = "".join(text_parts)
        
        return doc_id, full_text, all_annotations
    
    def convert(self):
        """Convert GENIA corpus XML files to BRAT format."""
        writer = BRATWriter(self.output_dir)
        
        xml_files = list(self.input_dir.glob("*.xml"))
        
        if not xml_files:
            logger.warning(f"No XML files found in {self.input_dir}")
            return
        
        logger.info(f"Converting {len(xml_files)} GENIA XML files...")
        
        total_articles = 0
        for xml_file in xml_files:
            logger.info(f"Processing {xml_file.name}...")
            
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
            except Exception as e:
                logger.error(f"Failed to parse {xml_file}: {e}")
                continue
            
            # Find all articles in the set
            articles = root.findall(".//article")
            
            for article in articles:
                try:
                    doc_id, text, annotations = self.convert_article(article)
                    
                    # Make doc_id unique by appending article index
                    unique_doc_id = f"{xml_file.stem}_{doc_id}_{total_articles}"
                    
                    writer.write_document(unique_doc_id, text, annotations)
                    total_articles += 1
                    
                except Exception as e:
                    logger.error(f"Failed to convert article in {xml_file}: {e}")
                    continue
        
        logger.info(f"Converted {total_articles} GENIA articles to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare NER datasets: convert to BRAT standoff format"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cadec", "share2013", "share2014", "genia"],
        help="Dataset to convert"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="../data/datasets/original",
        help="Base directory for original datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/datasets/prepared",
        help="Base directory for prepared datasets"
    )
    parser.add_argument(
        "--entity-types",
        type=str,
        nargs="+",
        default=None,
        help="Entity types to extract (CADEC only, default: ADR)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    input_base = Path(args.input_dir)
    output_base = Path(args.output_dir)
    
    input_dir = input_base / args.dataset
    output_dir = output_base / args.dataset
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    logger.info(f"Starting conversion: {args.dataset}")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    
    # Convert based on dataset
    try:
        if args.dataset == "cadec":
            converter = CADECConverter(
                input_dir, 
                output_dir,
                entity_types=args.entity_types
            )
            converter.convert()
        
        elif args.dataset == "share2013":
            converter = SHARE2013Converter(input_dir, output_dir)
            converter.convert()
        
        elif args.dataset == "share2014":
            converter = SHARE2014Converter(input_dir, output_dir)
            converter.convert()
        
        elif args.dataset == "genia":
            converter = GENIAConverter(input_dir, output_dir)
            converter.convert()
        
        logger.info(f"Converted {args.dataset}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()