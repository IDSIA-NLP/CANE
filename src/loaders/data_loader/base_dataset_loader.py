import json
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, DatasetDict, get_dataset_config_names
from loguru import logger
import time

class BaseDatasetLoader:
    def __init__(self, dataset_name, resource_path, examples_file, labels_file):
        self.dataset_name = dataset_name
        self.resource_path = resource_path
        self.examples_file = examples_file
        self.labels_file = labels_file

    def load_examples(self):
        """ Load examples from JSON files. """
        with open(self.resource_path + self.examples_file) as json_file:
            self.data_examples = json.load(json_file)
        logger.debug(f"Loaded examples for dataset.")

    def load_labels(self):
        """ Load labels from JSON file. """
        with open(self.resource_path + self.labels_file) as json_file:
            self.data_labels = json.load(json_file)
        logger.debug(f"Loaded labels for dataset.")

    def select_examples_with_empty(self, examples_list, num_examples, seed):
        """Sample examples, including one with no entities when num_examples > 1."""
        import random
        random.seed(seed)
        
        if num_examples <= 1 or len(examples_list) <= num_examples:
            # Not enough examples or only need 1, use simple sampling
            return random.sample(examples_list, min(num_examples, len(examples_list)))
        
        # Separate examples by entity presence
        with_entities = [ex for ex in examples_list if len(ex.get('entities', [])) > 0]
        without_entities = [ex for ex in examples_list if len(ex.get('entities', [])) == 0]
        
        selected = []
        
        # Try to include 1 zero-entity example
        if without_entities:
            selected.append(random.choice(without_entities))
            remaining_slots = num_examples - 1
        else:
            # No zero-entity examples available, use all with entities
            remaining_slots = num_examples
        
        # Fill remaining slots with entity examples
        if with_entities:
            num_to_select = min(remaining_slots, len(with_entities))
            selected.extend(random.sample(with_entities, num_to_select))
        
        return selected

    def load_hf_data(self, use_subset=True):
        """Load the dataset from the Hugging Face hub, merging subsets unless one is set in labels.json."""
        logger.debug(f"Loading dataset: {self.dataset_name}")
        start_time = time.time()

        # Strip language/variant suffix (e.g. "unimelb-nlp/wikiann:de" -> "unimelb-nlp/wikiann")
        hf_name = self.dataset_name.split(":")[0]
        
        subsets = get_dataset_config_names(hf_name, trust_remote_code=True)
        
        if len(subsets) > 1 and use_subset:
            # Load labels to check if a specific subset is specified
            if not hasattr(self, 'data_labels'):
                self.load_labels()
            
            # Check if 'sup' key exists in labels.json for this dataset
            specified_subset = None
            if self.dataset_name in self.data_labels:
                specified_subset = self.data_labels[self.dataset_name].get('sup', None)
            
            if specified_subset:
                # Load only the specified subset
                logger.debug(f"Loading specified subset: {specified_subset}")
                dataset = load_dataset(hf_name, specified_subset, trust_remote_code=True)
            else:
                # No subset specified - load all and concatenate (original behavior)
                logger.debug(f"No subset specified, loading all subsets: {subsets}")
                datasets_by_subset = {}

                for subset in subsets:
                    dataset_subset = load_dataset(hf_name, subset, download_mode="force_redownload", trust_remote_code=True)
                    
                    for split in dataset_subset.keys():
                        dataset_subset[split] = dataset_subset[split].add_column("subset", [subset] * len(dataset_subset[split]))
                        try:
                            datasets_by_subset[split].append(dataset_subset[split])
                        except KeyError:
                            logger.debug(f"Creating new key for split: {split}")
                            datasets_by_subset[split] = [dataset_subset[split]]

                concatenated_datasets = {}
                for split in datasets_by_subset.keys():
                    concatenated_datasets[split] = concatenate_datasets(datasets_by_subset[split])

                dataset = DatasetDict(concatenated_datasets)
        else:
            logger.debug(f"Loading dataset without subsets")
            dataset = load_dataset(hf_name, trust_remote_code=True)

        logger.debug(f"Time loading dataset--- {time.time() - start_time} seconds ---") 
        return dataset

    def process_labels(self, label_to_id, use_semantic_labels):
        """ Process and map labels if semantic labels are enabled. """
        if use_semantic_labels:
            label_to_semantic_label = self.data_labels[self.dataset_name]['label_to_semantic_label']
            label_to_id = {
                k if k == 'O' else k.split('-')[0] + '-' + label_to_semantic_label["-".join(k.split('-')[1:])]: v
                for k, v in label_to_id.items()
            }
        id_to_label = {i: label for label, i in label_to_id.items()}
        entities = list(label_to_id.keys())
        return label_to_id, id_to_label, entities

    def extract_base_entity_types(self, label_to_id):
        """Return the sorted entity types from IOB labels, without B-/I- prefixes."""
        base_types = set()
        for label in label_to_id.keys():
            if label == 'O':
                continue
            # Strip B-/I- prefix if present
            if '-' in label:
                base_type = label.split('-', 1)[1]
            else:
                base_type = label
            base_types.add(base_type)
        return sorted(list(base_types))

    def apply_semantic_labels_to_entities(self, entities_list, use_semantic_labels):
        """Return copies of the entity dicts with labels mapped to semantic labels."""
        if not use_semantic_labels:
            return entities_list
        
        # Load label mapping if not already loaded
        if not hasattr(self, 'data_labels'):
            self.load_labels()
        
        if self.dataset_name not in self.data_labels:
            return entities_list
        
        label_to_semantic = self.data_labels[self.dataset_name].get('label_to_semantic_label', {})
        
        if not label_to_semantic:
            return entities_list
        
        # Convert entity labels
        converted_entities = []
        for entity in entities_list:
            entity_copy = entity.copy()
            original_label = entity_copy['label']
            
            # Map label to semantic label if exists
            if original_label in label_to_semantic:
                entity_copy['label'] = label_to_semantic[original_label]
            
            converted_entities.append(entity_copy)
        
        return converted_entities

    def get_label_mapping(self, use_semantic_labels):
        """Return the original-to-semantic label mapping, or {} if not used."""
        if not use_semantic_labels:
            return {}
        
        # Load labels if not already loaded
        if not hasattr(self, 'data_labels'):
            self.load_labels()
        
        if self.dataset_name not in self.data_labels:
            return {}
        
        label_to_semantic = self.data_labels[self.dataset_name].get('label_to_semantic_label', {})
        return label_to_semantic

    def iob_to_char_offsets(self, tokens, iob_tags):
        """Convert IOB tags to (text, entities), with text joined on spaces and character spans."""
        # Reconstruct text
        text = " ".join(tokens)
        
        # Track character positions
        entities = []
        current_entity = None
        char_offset = 0
        
        for token, tag in zip(tokens, iob_tags):
            token_start = char_offset
            token_end = char_offset + len(token)
            
            if tag == 'O':
                # End current entity if exists
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            
            elif tag.startswith('B-'):
                # End previous entity if exists
                if current_entity:
                    entities.append(current_entity)
                
                # Start new entity
                label = tag[2:]  # Remove 'B-' prefix
                current_entity = {
                    'spans': [(token_start, token_end)],
                    'label': label,
                    'text': token
                }
            
            elif tag.startswith('I-'):
                # Continue current entity
                label = tag[2:]  # Remove 'I-' prefix
                
                if current_entity and current_entity['label'] == label:
                    # Extend current entity
                    current_entity['spans'][0] = (current_entity['spans'][0][0], token_end)
                    current_entity['text'] += " " + token
                else:
                    # Start new entity (handle malformed IOB)
                    if current_entity:
                        entities.append(current_entity)
                    
                    current_entity = {
                        'spans': [(token_start, token_end)],
                        'label': label,
                        'text': token
                    }
            
            # Move to next token (add 1 for space)
            char_offset = token_end + 1
        
        # Add final entity if exists
        if current_entity:
            entities.append(current_entity)
        
        return text, entities

    def spans_to_iob_tags(self, text, entities):
        """Convert character spans to (tokens, iob_tags) over whitespace tokens."""
        # Tokenize on whitespace
        tokens = text.split()
        iob_tags = ['O'] * len(tokens)
        
        # Build token offsets
        token_offsets = []
        char_offset = 0
        for token in tokens:
            start = text.find(token, char_offset)
            end = start + len(token)
            token_offsets.append((start, end))
            char_offset = end
        
        # Map entities to tokens
        for entity in entities:
            label = entity['label']
            # Handle discontinuous spans (only first span for IOB)
            entity_start, entity_end = entity['spans'][0]
            
            # Find overlapping tokens
            first_token_idx = None
            last_token_idx = None
            
            for idx, (tok_start, tok_end) in enumerate(token_offsets):
                # Check if token overlaps with entity span
                if tok_start < entity_end and tok_end > entity_start:
                    if first_token_idx is None:
                        first_token_idx = idx
                    last_token_idx = idx
            
            # Assign IOB tags
            if first_token_idx is not None:
                iob_tags[first_token_idx] = f'B-{label}'
                for idx in range(first_token_idx + 1, last_token_idx + 1):
                    iob_tags[idx] = f'I-{label}'
        
        return tokens, iob_tags

    def load_brat_data(self, brat_dir: Path, num_samples: int = None, seed: int = 42, label_mapping: dict = None):
        """Load .txt/.ann pairs from a BRAT directory as dicts with 'text' and 'entities'."""
        brat_dir = Path(brat_dir)
        txt_files = sorted(brat_dir.glob("*.txt"))
        
        if not txt_files:
            logger.warning(f"No .txt files found in {brat_dir}")
            return []
        
        label_mapping = label_mapping or {}
        
        data = []
        for txt_file in txt_files:
            ann_file = txt_file.with_suffix('.ann')
            
            # Read text
            with open(txt_file, 'r', encoding='utf-8', newline='') as f:
                text = f.read()
            
            # Read annotations
            entities = []
            if ann_file.exists():
                with open(ann_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line[0] != 'T':
                            continue
                        
                        parts = line.split('\t')
                        if len(parts) != 3:
                            continue
                        
                        # Parse: T1\tLabel start end;start end\ttext
                        ann_id, type_span, entity_text = parts
                        type_parts = type_span.split(' ', 1)
                        label = type_parts[0]
                        
                        # Apply label mapping if provided
                        mapped_label = label_mapping.get(label, label)
                        
                        # Parse spans (handle discontinuous with semicolon)
                        span_str = type_parts[1] if len(type_parts) > 1 else ""
                        spans = []
                        
                        for span_part in span_str.split(';'):
                            offsets = span_part.strip().split()
                            if len(offsets) == 2:
                                start, end = int(offsets[0]), int(offsets[1])
                                spans.append((start, end))
                        
                        if spans:
                            entities.append({
                                'spans': spans,
                                'label': mapped_label,
                                'text': entity_text
                            })
            
            data.append({
                'text': text,
                'entities': entities
            })
        
        # Sample if requested
        if num_samples and num_samples < len(data):
            import random
            random.seed(seed)
            data = random.sample(data, num_samples)
        
        return data