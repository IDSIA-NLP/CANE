from loaders.data_loader.ner_dataset_loaders import (
    Conll2003Loader, NcbiDiseaseLoader, WNUT2017Loader, BioNLP2004Loader, 
    BC5CDRLoader, FinLoader, Ontonotes5Loader,
    WikiNeuralLoader, MultiNERDLoader, CADECLoader, SHARE2013Loader, 
    SHARE2014Loader, SHARE2013ParagraphsLoader, SHARE2014ParagraphsLoader, GENIALoader, WikiAnnLoader
)

def get_dataset_and_entities(dataset_name, task, seed, num_test_samples, num_examples, use_semantic_labels, resource_path, examples_file, labels_file):
    if dataset_name == "conll2003":
        loader = Conll2003Loader(dataset_name, resource_path, examples_file, labels_file)
        return loader.get_data_and_entities(seed, num_test_samples, num_examples, use_semantic_labels)
    elif dataset_name == "ncbi_disease":
        loader = NcbiDiseaseLoader(dataset_name, resource_path, examples_file, labels_file)
        return loader.get_data_and_entities(seed, num_test_samples, num_examples, use_semantic_labels)
    elif dataset_name == "tner/wnut2017":
        loader = WNUT2017Loader(dataset_name, resource_path, examples_file, labels_file)
        return loader.get_data_and_entities(seed, num_test_samples, num_examples, use_semantic_labels)
    elif dataset_name == "tner/bionlp2004":
        loader = BioNLP2004Loader(dataset_name, resource_path, examples_file, labels_file)
        return loader.get_data_and_entities(seed, num_test_samples, num_examples, use_semantic_labels)
    elif dataset_name == "tner/bc5cdr":
        loader = BC5CDRLoader(dataset_name, resource_path, examples_file, labels_file)
        return loader.get_data_and_entities(seed, num_test_samples, num_examples, use_semantic_labels)
    elif dataset_name == "tner/fin":
        loader = FinLoader(dataset_name, resource_path, examples_file, labels_file)
        return loader.get_data_and_entities(seed, num_test_samples, num_examples, use_semantic_labels)
    elif dataset_name == "tner/ontonotes5":
        loader = Ontonotes5Loader(dataset_name, resource_path, examples_file, labels_file)
        return loader.get_data_and_entities(seed, num_test_samples, num_examples, use_semantic_labels)
    elif dataset_name == "tner/wikineural":
        loader = WikiNeuralLoader(dataset_name, resource_path, examples_file, labels_file)
        return loader.get_data_and_entities(seed, num_test_samples, num_examples, use_semantic_labels)
    elif dataset_name == "tner/multinerd":
        loader = MultiNERDLoader(dataset_name, resource_path, examples_file, labels_file)
        return loader.get_data_and_entities(seed, num_test_samples, num_examples, use_semantic_labels)
    elif dataset_name in ("unimelb-nlp/wikiann", "unimelb-nlp/wikiann:de", "unimelb-nlp/wikiann:fr", "unimelb-nlp/wikiann:zh"):
        loader = WikiAnnLoader(dataset_name, resource_path, examples_file, labels_file)
        return loader.get_data_and_entities(seed, num_test_samples, num_examples, use_semantic_labels)
    elif dataset_name == "cadec":
        loader = CADECLoader(dataset_name, resource_path, examples_file, labels_file)
        return loader.get_data_and_entities(seed, num_test_samples, num_examples, use_semantic_labels)
    elif dataset_name == "share2013":
        loader = SHARE2013Loader(dataset_name, resource_path, examples_file, labels_file)
        return loader.get_data_and_entities(seed, num_test_samples, num_examples, use_semantic_labels)
    elif dataset_name == "share2014":
        loader = SHARE2014Loader(dataset_name, resource_path, examples_file, labels_file)
        return loader.get_data_and_entities(seed, num_test_samples, num_examples, use_semantic_labels)
    elif dataset_name == "share2013_paragraphs":
        loader = SHARE2013ParagraphsLoader(dataset_name, resource_path, examples_file, labels_file)
        return loader.get_data_and_entities(seed, num_test_samples, num_examples, use_semantic_labels)
    elif dataset_name == "share2014_paragraphs":
        loader = SHARE2014ParagraphsLoader(dataset_name, resource_path, examples_file, labels_file)
        return loader.get_data_and_entities(seed, num_test_samples, num_examples, use_semantic_labels)
    elif dataset_name == "genia":
        loader = GENIALoader(dataset_name, resource_path, examples_file, labels_file)
        return loader.get_data_and_entities(seed, num_test_samples, num_examples, use_semantic_labels)
    else:
        raise NotImplementedError(f"NER dataset {dataset_name} not implemented.")