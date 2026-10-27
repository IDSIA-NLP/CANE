from loaders.config_loader import load_config
import os
from decouple import config as config_decouple

# Set HuggingFace token
os.environ['HF_TOKEN'] = config_decouple('HF_TOKEN')

# Load config and set cache directories FIRST
config = load_config("./config.yml")
if 'cache_dir' in config:
    cache = config['cache_dir']
    os.environ['HF_HOME'] = os.path.join(cache, 'huggingface')
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache, 'huggingface')
    os.environ['HF_DATASETS_CACHE'] = os.path.join(cache, 'huggingface/datasets')
    os.environ['XDG_CACHE_HOME'] = cache

from datasets import load_dataset, concatenate_datasets, DatasetDict, get_dataset_config_names
import json
from loguru import logger
from pathlib import Path

# Configure logging
logger.add("../logs/create_dataset_examples.log", format="{time} {level} {message}", level="INFO")

# Discontinuous ablation: prefer discontinuous examples and write examples_disco_ablation.json
DISCO_ABLATION = False

# Datasets used in the discontinuous ablation
DISCO_ABLATION_DATASETS = {"share2013_paragraphs", "share2014_paragraphs", "cadec"}


def iob_to_char_offsets(tokens, iob_tags):
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


def load_and_process_hf_dataset(dataset_name, label_to_id, min_unique_labels=5, max_examples=10, tag_column='ner_tags', subset=None):
    """
    Pick span-based examples from a Hugging Face dataset.

    Includes one zero-entity example if available; the rest need at least
    min_unique_labels distinct tags.
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Load the dataset
    try:
        if subset:
            dataset = load_dataset(dataset_name, subset)
        else:
            subsets = get_dataset_config_names(dataset_name)
            if len(subsets) > 1:
                logger.debug(f"Loading dataset with subsets: {subsets}")
                datasets_by_subset = {}

                for sub in subsets:
                    dataset_subset = load_dataset(dataset_name, sub, download_mode="force_redownload")
                    
                    for split in dataset_subset.keys():
                        dataset_subset[split] = dataset_subset[split].add_column("subset", [sub] * len(dataset_subset[split]))
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
                dataset = load_dataset(dataset_name)
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        return []
    
    # Reverse label mapping
    id_to_label = {i: label for label, i in label_to_id.items()}
    
    # Initialize examples storage
    examples_with_entities = []
    examples_without_entities = []
    
    # Find available split (try train, then validation, then test)
    available_split = None
    for split_name in ['train', 'validation', 'dev', 'test']:
        if split_name in dataset:
            available_split = split_name
            break
    
    if not available_split:
        logger.warning(f"No suitable split found for {dataset_name}. Available: {list(dataset.keys())}")
        return []
    
    logger.info(f"Using split '{available_split}' for {dataset_name}")
    
    # Iterate through dataset and process examples
    logger.info(f"Processing dataset: {dataset_name} with min_unique_labels={min_unique_labels} and max_examples={max_examples}")
    
    for i in range(len(dataset[available_split])):
        unique_tags = set(dataset[available_split][i][tag_column])
        
        tokens = dataset[available_split][i]['tokens']
        iob_tags = [id_to_label[tag_id] for tag_id in dataset[available_split][i][tag_column]]
        
        # Convert to span-based format
        text, entities = iob_to_char_offsets(tokens, iob_tags)
        
        example = {
            'text': text,
            'entities': entities
        }
        
        # Separate examples with and without entities
        if len(entities) == 0:
            examples_without_entities.append(example)
            logger.info(f"Found zero-entity example for {dataset_name}")
        elif len(unique_tags) >= min_unique_labels:
            examples_with_entities.append(example)
        
        # Stop when we have enough examples
        if len(examples_with_entities) >= max_examples and len(examples_without_entities) >= 1:
            break

    logger.info(f"Finished processing dataset: {dataset_name}. With entities: {len(examples_with_entities)}, Without entities: {len(examples_without_entities)}")
    
    # Combine examples: ensure at least 1 zero-entity example if available
    final_examples = []
    
    if examples_without_entities:
        # Add 1 zero-entity example
        final_examples.append(examples_without_entities[0])
        remaining_slots = max_examples - 1
    else:
        remaining_slots = max_examples
    
    # Fill remaining slots with entity examples
    final_examples.extend(examples_with_entities[:remaining_slots])
    
    logger.info(f"Selected {len(final_examples)} examples for {dataset_name} (including {1 if examples_without_entities else 0} zero-entity)")
    
    return final_examples


def _has_discontinuous(entities):
    """Check if any entity has multiple spans (discontinuous entity)."""
    return any(len(entity['spans']) > 1 for entity in entities)


def _has_nested(entities):
    """Check if any two entities have a fully nested span relationship."""
    for i, e1 in enumerate(entities):
        for j, e2 in enumerate(entities):
            if i == j:
                continue
            # Use first span of each entity for nesting check
            s1, e1_end = e1['spans'][0]
            s2, e2_end = e2['spans'][0]
            # Check if e2 is fully contained within e1
            if s1 <= s2 and e1_end >= e2_end:
                return True
    return False


def load_and_process_brat_dataset(dataset_name, data_dir, max_examples=10, discontinuous_only=False):
    """
    Pick examples from a BRAT dataset.

    By default takes one each of regular, zero-entity, discontinuous and nested
    (in that order, when available) and fills the rest with regular examples.
    With discontinuous_only=True, takes one zero-entity example and fills the
    rest with examples containing a discontinuous entity, falling back to
    regular ones.
    """
    logger.info(f"Loading BRAT dataset: {dataset_name} from {data_dir}")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"BRAT dataset directory not found: {data_dir}")
        return []
    
    # Four buckets in priority order: regular, zero, discontinuous, nested
    examples_regular = []
    examples_without_entities = []
    examples_discontinuous = []
    examples_nested = []

    txt_files = sorted(data_path.glob("*.txt"))
    
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
                            'label': label,
                            'text': entity_text
                        })
        
        example = {
            'text': text,
            'entities': entities
        }
        
        # Assign example to exactly one bucket (rarest special bucket takes priority)
        if len(entities) == 0:
            examples_without_entities.append(example)
            logger.info(f"Found zero-entity example in BRAT {dataset_name}")
        elif _has_nested(entities):
            # Nested wins even if also discontinuous
            examples_nested.append(example)
            logger.info(f"Found nested-entity example in BRAT {dataset_name}")
        elif _has_discontinuous(entities):
            examples_discontinuous.append(example)
            logger.info(f"Found discontinuous-entity example in BRAT {dataset_name}")
        else:
            examples_regular.append(example)
        
        # Stop when we have enough across all buckets
        if (len(examples_regular) >= max_examples and
                len(examples_without_entities) >= 1 and
                len(examples_discontinuous) >= 1 and
                len(examples_nested) >= 1):
            break
    
    logger.info(
        f"Finished processing BRAT dataset: {dataset_name}. "
        f"Regular: {len(examples_regular)}, Zero-entity: {len(examples_without_entities)}, "
        f"Discontinuous: {len(examples_discontinuous)}, Nested: {len(examples_nested)}"
    )

    # Discontinuous ablation selection
    if discontinuous_only:
        # Discontinuous files first, then nested files that also have a discontinuous entity
        disco_pool = examples_discontinuous + [
            e for e in examples_nested if _has_discontinuous(e['entities'])
        ]

        final_examples = []
        remaining_slots = max_examples

        # Priority 1: zero-entity example
        if examples_without_entities and remaining_slots > 0:
            final_examples.append(examples_without_entities[0])
            remaining_slots -= 1

        # Fill remaining slots with discontinuous examples
        disco_taken = disco_pool[:remaining_slots]
        final_examples.extend(disco_taken)
        remaining_slots -= len(disco_taken)

        # Fall back to regular examples if disco pool was exhausted
        if remaining_slots > 0:
            logger.warning(
                f"Disco pool exhausted for {dataset_name}; "
                f"falling back to {remaining_slots} regular example(s)."
            )
            final_examples.extend(examples_regular[:remaining_slots])

        logger.info(
            f"Selected {len(final_examples)} examples for {dataset_name} "
            f"[disco_ablation=True] "
            f"(zero-entity: {1 if examples_without_entities else 0}, "
            f"discontinuous: {len(disco_taken)}, "
            f"regular fallback: {max(0, max_examples - 1 - len(disco_taken))})"
        )
        return final_examples

    # Standard selection
    final_examples = []
    remaining_slots = max_examples

    # Priority 1: regular entity example, so a normal example comes first
    if examples_regular and remaining_slots > 0:
        final_examples.append(examples_regular[0])
        remaining_slots -= 1

    # Priority 2: zero-entity example
    if examples_without_entities and remaining_slots > 0:
        final_examples.append(examples_without_entities[0])
        remaining_slots -= 1

    # Priority 3: discontinuous entity example
    if examples_discontinuous and remaining_slots > 0:
        final_examples.append(examples_discontinuous[0])
        remaining_slots -= 1

    # Priority 4: nested entity example
    if examples_nested and remaining_slots > 0:
        final_examples.append(examples_nested[0])
        remaining_slots -= 1

    # Fill remaining slots with additional regular entity examples
    final_examples.extend(examples_regular[1:remaining_slots + 1])
    
    logger.info(
        f"Selected {len(final_examples)} examples for {dataset_name} "
        f"(regular: {min(1, len(examples_regular))}, "
        f"zero-entity: {1 if examples_without_entities else 0}, "
        f"discontinuous: {1 if examples_discontinuous else 0}, "
        f"nested: {1 if examples_nested else 0})"
    )
    
    return final_examples


def save_examples_to_json(examples_dict, output_file):
    """Save examples to JSON file."""
    logger.info(f"Saving examples to JSON file: {output_file}")
    
    with open(output_file, 'w') as f:
        json.dump(examples_dict, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Successfully saved examples to {output_file}")


if __name__ == "__main__":

    # Output file reflects whether this is the disco ablation run
    output_file = '../resources/examples_disco_ablation.json' if DISCO_ABLATION else '../resources/examples.json'

    # Load the data config file json from the resources folder
    with open('../resources/labels.json') as file:
        datasets_config = json.load(file)

    # Filter to only NER datasets
    datasets_config = {k: v for k, v in datasets_config.items() if v.get('task') == 'ner'}
    
    all_examples = {}

    logger.info("Starting the example creation process")
    
    # Process HuggingFace datasets
    hf_datasets = [
        "conll2003",
        "ncbi_disease",
        "tner/bionlp2004",
        "tner/wnut2017",
        "tner/bc5cdr",
        "tner/fin",
        "tner/ontonotes5",
        "tner/wikineural",
        "tner/multinerd",
        "unimelb-nlp/wikiann",
        "unimelb-nlp/wikiann:de",
        "unimelb-nlp/wikiann:fr",
        "unimelb-nlp/wikiann:zh"
    ]
    
    for dataset_name in hf_datasets:
        if dataset_name not in datasets_config:
            logger.warning(f"Dataset {dataset_name} not in config, skipping")
            continue
        
        config = datasets_config[dataset_name]
        
        # Handle datasets with subsets
        subset = config.get('sup', None)

        # Strip language/variant suffix to get the actual HuggingFace dataset name
        hf_name = dataset_name.split(":")[0]
        
        try:
            examples = load_and_process_hf_dataset(
                hf_name,                        # use hf_name for loading
                label_to_id=config["label_to_id"],
                min_unique_labels=config["min_unique_labels"],
                max_examples=10,
                tag_column=config["tag_column"],
                subset=subset
            )
            
            if examples:
                all_examples[dataset_name] = examples  # keep original name as key

        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {e}")
    
    # Process BRAT datasets
    brat_datasets = {
        "cadec": "../data/datasets/prepared/cadec",
        "share2013": "../data/datasets/prepared/share2013/test",
        "share2014": "../data/datasets/prepared/share2014/test",
        "share2013_paragraphs": "../data/datasets/prepared/share2013_paragraphs/test",
        "share2014_paragraphs": "../data/datasets/prepared/share2014_paragraphs/test",
        "genia": "../data/datasets/prepared/genia"
    }
    
    for dataset_name, data_dir in brat_datasets.items():
        if dataset_name not in datasets_config:
            logger.warning(f"Dataset {dataset_name} not in config, skipping")
            continue

        # Use discontinuous-heavy selection for supported datasets when in ablation mode
        use_disco = DISCO_ABLATION and dataset_name in DISCO_ABLATION_DATASETS
        
        try:
            examples = load_and_process_brat_dataset(
                dataset_name,
                data_dir,
                max_examples=10,
                discontinuous_only=use_disco
            )
            
            if examples:
                all_examples[dataset_name] = examples
        except Exception as e:
            logger.error(f"Error processing BRAT dataset {dataset_name}: {e}")
        
    # Save to JSON
    save_examples_to_json(all_examples, output_file)

    # Load examples JSON file and print the first example
    with open(output_file) as file:
        json_data = json.load(file)

    logger.info(f"First example from {output_file}:")
    for dataset_name in json_data:
        if json_data[dataset_name]:
            logger.info(f"\nDataset: {dataset_name}")
            logger.info(f"Example: {json.dumps(json_data[dataset_name][0], indent=2, ensure_ascii=False)}")
            break
    
    logger.info("Done!")
