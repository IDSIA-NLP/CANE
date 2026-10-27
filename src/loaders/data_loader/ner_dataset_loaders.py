from loaders.data_loader.base_dataset_loader import BaseDatasetLoader
import random
from pathlib import Path

'''
The following classes load NER datasets and convert them to a unified format:
{
    'text': 'full sentence text',
    'entities': [
        {
            'spans': [(start_char, end_char)],  # character offsets
            'label': 'Disease',
            'text': 'entity text'
        }
    ]
}

For HuggingFace datasets: reconstruct text from tokens and convert IOB tags to character offsets.
For BRAT datasets: load directly from .txt and .ann files.
'''

class Conll2003Loader(BaseDatasetLoader):
    def get_data_and_entities(self, seed, num_test_samples, num_examples, use_semantic_labels):
        self.load_examples()
        self.load_labels()
        dataset = self.load_hf_data()
        label_to_id = self.data_labels[self.dataset_name]['label_to_id']
        label_to_id, id_to_label, entities = self.process_labels(label_to_id, use_semantic_labels)
        
        # Get test data
        actual_size = min(num_test_samples, len(dataset['test']))
        test_dataset = dataset['test'].shuffle(seed=seed).select(range(actual_size))

        test_data = []
        for row in test_dataset:
            tokens = row['tokens']
            iob_tags = [id_to_label[i] for i in row['ner_tags']]
            text, entities_list = self.iob_to_char_offsets(tokens, iob_tags)
            test_data.append({'text': text, 'entities': entities_list})
        
        # Get examples and apply semantic conversion
        examples_list = self.data_examples[self.dataset_name]
        examples_data = self.select_examples_with_empty(examples_list, num_examples, seed)
        
        # Apply semantic labels to examples
        for example in examples_data:
            example['entities'] = self.apply_semantic_labels_to_entities(example['entities'], use_semantic_labels)
        
        new_dataset = {
            'test': test_data,
            'examples': examples_data
        }
        return new_dataset, self.extract_base_entity_types(label_to_id)


class NcbiDiseaseLoader(BaseDatasetLoader):
    def get_data_and_entities(self, seed, num_test_samples, num_examples, use_semantic_labels):
        self.load_examples()
        self.load_labels()
        dataset = self.load_hf_data()
        label_to_id = self.data_labels[self.dataset_name]['label_to_id']
        label_to_id, id_to_label, entities = self.process_labels(label_to_id, use_semantic_labels)
        
        # Get test data
        actual_size = min(num_test_samples, len(dataset['test']))
        test_dataset = dataset['test'].shuffle(seed=seed).select(range(actual_size))

        test_data = []
        for row in test_dataset:
            tokens = row['tokens']
            iob_tags = [id_to_label[i] for i in row['ner_tags']]
            text, entities_list = self.iob_to_char_offsets(tokens, iob_tags)
            test_data.append({'text': text, 'entities': entities_list})
        
        # Get examples and apply semantic conversion
        examples_list = self.data_examples[self.dataset_name]
        examples_data = self.select_examples_with_empty(examples_list, num_examples, seed)
        
        # Apply semantic labels to examples
        for example in examples_data:
            example['entities'] = self.apply_semantic_labels_to_entities(example['entities'], use_semantic_labels)
        
        new_dataset = {
            'test': test_data,
            'examples': examples_data
        }
        return new_dataset, self.extract_base_entity_types(label_to_id)


class WNUT2017Loader(BaseDatasetLoader):
    def get_data_and_entities(self, seed, num_test_samples, num_examples, use_semantic_labels):
        self.load_examples()
        self.load_labels()
        dataset = self.load_hf_data()
        label_to_id = self.data_labels[self.dataset_name]['label_to_id']
        label_to_id, id_to_label, entities = self.process_labels(label_to_id, use_semantic_labels)
        
        # Get test data
        actual_size = min(num_test_samples, len(dataset['test']))
        test_dataset = dataset['test'].shuffle(seed=seed).select(range(actual_size))
        
        test_data = []
        for row in test_dataset:
            tokens = row['tokens']
            iob_tags = [id_to_label[i] for i in row['tags']]
            text, entities_list = self.iob_to_char_offsets(tokens, iob_tags)
            test_data.append({'text': text, 'entities': entities_list})
        
        # Get examples and apply semantic conversion
        examples_list = self.data_examples[self.dataset_name]
        examples_data = self.select_examples_with_empty(examples_list, num_examples, seed)
        
        # Apply semantic labels to examples
        for example in examples_data:
            example['entities'] = self.apply_semantic_labels_to_entities(example['entities'], use_semantic_labels)
        
        new_dataset = {
            'test': test_data,
            'examples': examples_data
        }
        return new_dataset, self.extract_base_entity_types(label_to_id)


class BioNLP2004Loader(BaseDatasetLoader):
    def get_data_and_entities(self, seed, num_test_samples, num_examples, use_semantic_labels):
        self.load_examples()
        self.load_labels()
        dataset = self.load_hf_data()
        label_to_id = self.data_labels[self.dataset_name]['label_to_id']
        label_to_id, id_to_label, entities = self.process_labels(label_to_id, use_semantic_labels)
        
        # Get test data
        actual_size = min(num_test_samples, len(dataset['test']))
        test_dataset = dataset['test'].shuffle(seed=seed).select(range(actual_size))

        test_data = []
        for row in test_dataset:
            tokens = row['tokens']
            iob_tags = [id_to_label[i] for i in row['tags']]
            text, entities_list = self.iob_to_char_offsets(tokens, iob_tags)
            test_data.append({'text': text, 'entities': entities_list})
        
        # Get examples and apply semantic conversion
        examples_list = self.data_examples[self.dataset_name]
        examples_data = self.select_examples_with_empty(examples_list, num_examples, seed)
        
        # Apply semantic labels to examples
        for example in examples_data:
            example['entities'] = self.apply_semantic_labels_to_entities(example['entities'], use_semantic_labels)
        
        new_dataset = {
            'test': test_data,
            'examples': examples_data
        }
        return new_dataset, self.extract_base_entity_types(label_to_id)


class BC5CDRLoader(BaseDatasetLoader):
    def get_data_and_entities(self, seed, num_test_samples, num_examples, use_semantic_labels):
        self.load_examples()
        self.load_labels()
        dataset = self.load_hf_data()
        label_to_id = self.data_labels[self.dataset_name]['label_to_id']
        label_to_id, id_to_label, entities = self.process_labels(label_to_id, use_semantic_labels)
        
        # Get test data
        actual_size = min(num_test_samples, len(dataset['test']))
        test_dataset = dataset['test'].shuffle(seed=seed).select(range(actual_size))

        test_data = []
        for row in test_dataset:
            tokens = row['tokens']
            iob_tags = [id_to_label[i] for i in row['tags']]
            text, entities_list = self.iob_to_char_offsets(tokens, iob_tags)
            test_data.append({'text': text, 'entities': entities_list})
        
        # Get examples and apply semantic conversion
        examples_list = self.data_examples[self.dataset_name]
        random.seed(seed)
        examples_data = random.sample(examples_list, min(num_examples, len(examples_list)))
        
        # Apply semantic labels to examples
        for example in examples_data:
            example['entities'] = self.apply_semantic_labels_to_entities(example['entities'], use_semantic_labels)
        
        new_dataset = {
            'test': test_data,
            'examples': examples_data
        }
        return new_dataset, self.extract_base_entity_types(label_to_id)


class FinLoader(BaseDatasetLoader):
    def get_data_and_entities(self, seed, num_test_samples, num_examples, use_semantic_labels):
        self.load_examples()
        self.load_labels()
        dataset = self.load_hf_data()
        label_to_id = self.data_labels[self.dataset_name]['label_to_id']
        label_to_id, id_to_label, entities = self.process_labels(label_to_id, use_semantic_labels)
        
        # Get test data
        actual_size = min(num_test_samples, len(dataset['test']))
        test_dataset = dataset['test'].shuffle(seed=seed).select(range(actual_size))

        test_data = []
        for row in test_dataset:
            tokens = row['tokens']
            iob_tags = [id_to_label[i] for i in row['tags']]
            text, entities_list = self.iob_to_char_offsets(tokens, iob_tags)
            test_data.append({'text': text, 'entities': entities_list})
        
        # Get examples and apply semantic conversion
        examples_list = self.data_examples[self.dataset_name]
        examples_data = self.select_examples_with_empty(examples_list, num_examples, seed)
        
        # Apply semantic labels to examples
        for example in examples_data:
            example['entities'] = self.apply_semantic_labels_to_entities(example['entities'], use_semantic_labels)
        
        new_dataset = {
            'test': test_data,
            'examples': examples_data
        }
        return new_dataset, self.extract_base_entity_types(label_to_id)


class Ontonotes5Loader(BaseDatasetLoader):
    def get_data_and_entities(self, seed, num_test_samples, num_examples, use_semantic_labels):
        self.load_examples()
        self.load_labels()
        dataset = self.load_hf_data()
        label_to_id = self.data_labels[self.dataset_name]['label_to_id']
        label_to_id, id_to_label, entities = self.process_labels(label_to_id, use_semantic_labels)
        
        # Get test data
        actual_size = min(num_test_samples, len(dataset['test']))
        test_dataset = dataset['test'].shuffle(seed=seed).select(range(actual_size))

        test_data = []
        for row in test_dataset:
            tokens = row['tokens']
            iob_tags = [id_to_label[i] for i in row['tags']]
            text, entities_list = self.iob_to_char_offsets(tokens, iob_tags)
            test_data.append({'text': text, 'entities': entities_list})
        
        # Get examples and apply semantic conversion
        examples_list = self.data_examples[self.dataset_name]
        examples_data = self.select_examples_with_empty(examples_list, num_examples, seed)
        
        # Apply semantic labels to examples
        for example in examples_data:
            example['entities'] = self.apply_semantic_labels_to_entities(example['entities'], use_semantic_labels)
        
        new_dataset = {
            'test': test_data,
            'examples': examples_data
        }
        return new_dataset, self.extract_base_entity_types(label_to_id)


class WikiNeuralLoader(BaseDatasetLoader):
    def get_data_and_entities(self, seed, num_test_samples, num_examples, use_semantic_labels):
        self.load_examples()
        self.load_labels()
        dataset = self.load_hf_data()
        label_to_id = self.data_labels[self.dataset_name]['label_to_id']
        label_to_id, id_to_label, entities = self.process_labels(label_to_id, use_semantic_labels)
        
        # Get test data
        actual_size = min(num_test_samples, len(dataset['test']))
        test_dataset = dataset['test'].shuffle(seed=seed).select(range(actual_size))

        test_data = []
        for row in test_dataset:
            tokens = row['tokens']
            iob_tags = [id_to_label[i] for i in row['tags']]
            text, entities_list = self.iob_to_char_offsets(tokens, iob_tags)
            test_data.append({'text': text, 'entities': entities_list})
        
        # Get examples and apply semantic conversion
        examples_list = self.data_examples[self.dataset_name]
        examples_data = self.select_examples_with_empty(examples_list, num_examples, seed)
        
        # Apply semantic labels to examples
        for example in examples_data:
            example['entities'] = self.apply_semantic_labels_to_entities(example['entities'], use_semantic_labels)
        
        new_dataset = {
            'test': test_data,
            'examples': examples_data
        }
        return new_dataset, self.extract_base_entity_types(label_to_id)


class MultiNERDLoader(BaseDatasetLoader):
    def get_data_and_entities(self, seed, num_test_samples, num_examples, use_semantic_labels):
        self.load_examples()
        self.load_labels()
        dataset = self.load_hf_data()
        label_to_id = self.data_labels[self.dataset_name]['label_to_id']
        label_to_id, id_to_label, entities = self.process_labels(label_to_id, use_semantic_labels)
        
        # Get test data
        actual_size = min(num_test_samples, len(dataset['test']))
        test_dataset = dataset['test'].shuffle(seed=seed).select(range(actual_size))

        test_data = []
        for row in test_dataset:
            tokens = row['tokens']
            iob_tags = [id_to_label[i] for i in row['tags']]
            text, entities_list = self.iob_to_char_offsets(tokens, iob_tags)
            test_data.append({'text': text, 'entities': entities_list})
        
        # Get examples and apply semantic conversion
        examples_list = self.data_examples[self.dataset_name]
        examples_data = self.select_examples_with_empty(examples_list, num_examples, seed)
        
        # Apply semantic labels to examples
        for example in examples_data:
            example['entities'] = self.apply_semantic_labels_to_entities(example['entities'], use_semantic_labels)
        
        new_dataset = {
            'test': test_data,
            'examples': examples_data
        }
        return new_dataset, self.extract_base_entity_types(label_to_id)


class WikiAnnLoader(BaseDatasetLoader):
    def get_data_and_entities(self, seed, num_test_samples, num_examples, use_semantic_labels):
        self.load_examples()
        self.load_labels()
        dataset = self.load_hf_data()  # Will use 'sup' from labels.json
        label_to_id = self.data_labels[self.dataset_name]['label_to_id']
        label_to_id, id_to_label, entities = self.process_labels(label_to_id, use_semantic_labels)
        
        # Process test data
        actual_size = min(num_test_samples, len(dataset['test']))
        test_dataset = dataset['test'].shuffle(seed=seed).select(range(actual_size))
        
        test_data = []
        for row in test_dataset:
            tokens = row['tokens']
            iob_tags = [id_to_label[i] for i in row['ner_tags']]
            text, entities_list = self.iob_to_char_offsets(tokens, iob_tags)
            test_data.append({'text': text, 'entities': entities_list})
        
        # Get examples
        examples_list = self.data_examples[self.dataset_name]
        examples_data = self.select_examples_with_empty(examples_list, num_examples, seed)
        
        # Apply semantic labels
        for example in examples_data:
            example['entities'] = self.apply_semantic_labels_to_entities(example['entities'], use_semantic_labels)
        
        new_dataset = {
            'test': test_data,
            'examples': examples_data
        }
        return new_dataset, self.extract_base_entity_types(label_to_id)
    

# BRAT-based loaders for complex datasets
class CADECLoader(BaseDatasetLoader):
    def get_data_and_entities(self, seed, num_test_samples, num_examples, use_semantic_labels):
        """Load CADEC from prepared BRAT files."""
        prepared_dir = Path("../data/datasets/prepared/cadec")
        
        if not prepared_dir.exists():
            raise ValueError(f"Prepared CADEC dataset not found at {prepared_dir}. Run prepare_datasets.py first.")
        
        # Load labels and examples
        self.load_labels()
        self.load_examples()
        
        # Get label mapping for semantic labels
        label_mapping = self.get_label_mapping(use_semantic_labels)
        
        # Get examples, including a zero-entity one
        examples_list = self.data_examples[self.dataset_name]
        examples_data = self.select_examples_with_empty(examples_list, num_examples, seed)
        
        # Apply semantic labels to examples
        for example in examples_data:
            example['entities'] = self.apply_semantic_labels_to_entities(example['entities'], use_semantic_labels)
        
        # Load test data from BRAT files
        all_test_data = self.load_brat_data(prepared_dir, label_mapping=label_mapping)
        
        # Exclude examples from test data to avoid overlap
        examples_texts = {ex['text'] for ex in examples_data}
        test_data = [item for item in all_test_data if item['text'] not in examples_texts]
        
        # Limit to requested number of test samples
        random.seed(seed)
        random.shuffle(test_data)
        test_data = test_data[:num_test_samples] if num_test_samples < len(test_data) else test_data
        
        # Extract entity types
        entity_types = set()
        for item in all_test_data:
            for entity in item['entities']:
                entity_types.add(entity['label'])
        
        new_dataset = {
            'test': test_data,
            'examples': examples_data
        }
        return new_dataset, sorted(list(entity_types))


class SHARE2013Loader(BaseDatasetLoader):
    def get_data_and_entities(self, seed, num_test_samples, num_examples, use_semantic_labels):
        """Load SHARE 2013 from prepared BRAT files."""
        prepared_dir = Path("../data/datasets/prepared/share2013")
        
        if not prepared_dir.exists():
            raise ValueError(f"Prepared SHARE 2013 dataset not found at {prepared_dir}. Run prepare_datasets.py first.")
        
        # Load labels and examples
        self.load_labels()
        self.load_examples()
        
        # Get label mapping for semantic labels
        label_mapping = self.get_label_mapping(use_semantic_labels)
        
        # Get examples, including a zero-entity one
        examples_list = self.data_examples[self.dataset_name]
        examples_data = self.select_examples_with_empty(examples_list, num_examples, seed)
        
        # Apply semantic labels to examples
        for example in examples_data:
            example['entities'] = self.apply_semantic_labels_to_entities(example['entities'], use_semantic_labels)
        
        # Load test data from BRAT files
        all_test_data = self.load_brat_data(prepared_dir / "test", label_mapping=label_mapping)
        
        # Exclude examples from test data to avoid overlap
        examples_texts = {ex['text'] for ex in examples_data}
        test_data = [item for item in all_test_data if item['text'] not in examples_texts]
        
        # Limit to requested number of test samples
        random.seed(seed)
        random.shuffle(test_data)
        test_data = test_data[:num_test_samples] if num_test_samples < len(test_data) else test_data
        
        # Extract entity types
        entity_types = set()
        for item in all_test_data + examples_data:
            for entity in item['entities']:
                entity_types.add(entity['label'])
        
        new_dataset = {
            'test': test_data,
            'examples': examples_data
        }
        return new_dataset, sorted(list(entity_types))


class SHARE2014Loader(BaseDatasetLoader):
    def get_data_and_entities(self, seed, num_test_samples, num_examples, use_semantic_labels):
        """Load SHARE 2014 from prepared BRAT files."""
        prepared_dir = Path("../data/datasets/prepared/share2014")
        
        if not prepared_dir.exists():
            raise ValueError(f"Prepared SHARE 2014 dataset not found at {prepared_dir}. Run prepare_datasets.py first.")
        
        # Load labels and examples
        self.load_labels()
        self.load_examples()
        
        # Get label mapping for semantic labels
        label_mapping = self.get_label_mapping(use_semantic_labels)
        
        # Get examples, including a zero-entity one
        examples_list = self.data_examples[self.dataset_name]
        examples_data = self.select_examples_with_empty(examples_list, num_examples, seed)
        
        # Apply semantic labels to examples
        for example in examples_data:
            example['entities'] = self.apply_semantic_labels_to_entities(example['entities'], use_semantic_labels)
        
        # Load test data from BRAT files
        all_test_data = self.load_brat_data(prepared_dir / "test", label_mapping=label_mapping)
        
        # Exclude examples from test data to avoid overlap
        examples_texts = {ex['text'] for ex in examples_data}
        test_data = [item for item in all_test_data if item['text'] not in examples_texts]
        
        # Limit to requested number of test samples
        random.seed(seed)
        random.shuffle(test_data)
        test_data = test_data[:num_test_samples] if num_test_samples < len(test_data) else test_data
        
        # Extract entity types
        entity_types = set()
        for item in all_test_data + examples_data:
            for entity in item['entities']:
                entity_types.add(entity['label'])
        
        new_dataset = {
            'test': test_data,
            'examples': examples_data
        }
        return new_dataset, sorted(list(entity_types))


class GENIALoader(BaseDatasetLoader):
    def get_data_and_entities(self, seed, num_test_samples, num_examples, use_semantic_labels):
        """Load GENIA from prepared BRAT files."""
        prepared_dir = Path("../data/datasets/prepared/genia")
        
        if not prepared_dir.exists():
            raise ValueError(f"Prepared GENIA dataset not found at {prepared_dir}. Run prepare_datasets.py first.")
        
        # Load labels and examples
        self.load_labels()
        self.load_examples()
        
        # Get label mapping for semantic labels
        label_mapping = self.get_label_mapping(use_semantic_labels)
        
        # Get examples, including a zero-entity one
        examples_list = self.data_examples[self.dataset_name]
        examples_data = self.select_examples_with_empty(examples_list, num_examples, seed)
        
        # Apply semantic labels to examples
        for example in examples_data:
            example['entities'] = self.apply_semantic_labels_to_entities(example['entities'], use_semantic_labels)
        
        # Load test data from BRAT files
        all_test_data = self.load_brat_data(prepared_dir, label_mapping=label_mapping)
        
        # Exclude examples from test data to avoid overlap
        examples_texts = {ex['text'] for ex in examples_data}
        test_data = [item for item in all_test_data if item['text'] not in examples_texts]
        
        # Limit to requested number of test samples
        random.seed(seed)
        random.shuffle(test_data)
        test_data = test_data[:num_test_samples] if num_test_samples < len(test_data) else test_data
        
        # Extract entity types
        entity_types = set()
        for item in all_test_data:
            for entity in item['entities']:
                entity_types.add(entity['label'])
        
        new_dataset = {
            'test': test_data,
            'examples': examples_data
        }
        return new_dataset, sorted(list(entity_types))
    

class SHARE2013ParagraphsLoader(SHARE2013Loader):
    def get_data_and_entities(self, seed, num_test_samples, num_examples, use_semantic_labels):
        prepared_dir = Path("../data/datasets/prepared/share2013_paragraphs")
        if not prepared_dir.exists():
            raise ValueError(f"Prepared SHARE 2013 paragraphs dataset not found at {prepared_dir}. Run chunk_share_datasets_paragraphs.py first.")
        self.load_labels()
        self.load_examples()
        label_mapping = self.get_label_mapping(use_semantic_labels)
        examples_list = self.data_examples["share2013_paragraphs"]
        examples_data = self.select_examples_with_empty(examples_list, num_examples, seed)
        for example in examples_data:
            example['entities'] = self.apply_semantic_labels_to_entities(example['entities'], use_semantic_labels)
        all_test_data = self.load_brat_data(prepared_dir / "test", label_mapping=label_mapping)
        examples_texts = {ex['text'] for ex in examples_data}
        test_data = [item for item in all_test_data if item['text'] not in examples_texts]
        random.seed(seed)
        random.shuffle(test_data)
        test_data = test_data[:num_test_samples] if num_test_samples < len(test_data) else test_data
        entity_types = set()
        for item in all_test_data + examples_data:
            for entity in item['entities']:
                entity_types.add(entity['label'])
        return {'test': test_data, 'examples': examples_data}, sorted(list(entity_types))


class SHARE2014ParagraphsLoader(SHARE2014Loader):
    def get_data_and_entities(self, seed, num_test_samples, num_examples, use_semantic_labels):
        prepared_dir = Path("../data/datasets/prepared/share2014_paragraphs")
        if not prepared_dir.exists():
            raise ValueError(f"Prepared SHARE 2014 paragraphs dataset not found at {prepared_dir}. Run chunk_share_datasets_paragraphs.py first.")
        self.load_labels()
        self.load_examples()
        label_mapping = self.get_label_mapping(use_semantic_labels)
        examples_list = self.data_examples["share2014_paragraphs"]
        examples_data = self.select_examples_with_empty(examples_list, num_examples, seed)
        for example in examples_data:
            example['entities'] = self.apply_semantic_labels_to_entities(example['entities'], use_semantic_labels)
        all_test_data = self.load_brat_data(prepared_dir / "test", label_mapping=label_mapping)
        examples_texts = {ex['text'] for ex in examples_data}
        test_data = [item for item in all_test_data if item['text'] not in examples_texts]
        random.seed(seed)
        random.shuffle(test_data)
        test_data = test_data[:num_test_samples] if num_test_samples < len(test_data) else test_data
        entity_types = set()
        for item in all_test_data + examples_data:
            for entity in item['entities']:
                entity_types.add(entity['label'])
        return {'test': test_data, 'examples': examples_data}, sorted(list(entity_types))