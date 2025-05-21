import argparse
import os
import re

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Value
from dotenv import find_dotenv, load_dotenv
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


MAX_TOKENS_IOB_SENT = 256
OVERLAPPING_LEN = 0

# Split text into sentence-like chunks (page_content only)
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name='o200k_base',
    separators=["\n\n\n", "\n\n", "\n", " .", " !", " ?", " ،", " ,", " ", ""],
    keep_separator=True,
    chunk_size=MAX_TOKENS_IOB_SENT,
    chunk_overlap=OVERLAPPING_LEN,
)

print(load_dotenv(find_dotenv(".env")))


def assign_iob_tags(text, token_spans, entities):
    """
    Assign IOB tags to tokens based on character-based entity spans.

    Args:
        text: Original text.
        token_spans: List of (start, end) character indices for tokens.
        entities: List of dicts with 'start', 'end', 'tag'.

    Returns:
        List of (token_text, tag) tuples.
    """
    tags = []

    for token in token_spans:
        token_start, token_end = token.start(), token.end()
        token_tag = 'O'
        for entity in entities:
            ent_start, ent_end, ent_tag = entity[0], entity[1], entity[2]
            if token_start >= ent_start and token_end <= ent_end:
                if token_start == ent_start:
                    token_tag = f'B-{ent_tag}'
                else:
                    token_tag = f'I-{ent_tag}'
                break
        token_text = text[token_start:token_end]
        tags.append(f"{token_text}\t{token_tag}")
    
    return tags


def split_sentence_with_indices(text):
    pattern = r'''
        (?:
            \d+[.,]?\d*\s*[%$€]?               # Numbers with optional decimal/currency
        )
        |
        \w+(?:-\w+)*                           # Words with optional hyphens (Unicode-aware)
        |
        [()\[\]{}]                             # Parentheses and brackets
        |
        [^\w\s]                                # Other single punctuation marks
    '''
    return list(re.finditer(pattern, text, flags=re.VERBOSE | re.UNICODE))

def chunk_iob_tagged_text(text, iob_data, token_spans):
    """
    Split long IOB-tagged text into sentence-like chunks using a character splitter,
    then align tokens and their IOB tags within each chunk.

    Parameters:
    - text: str, full input text
    - iob_data: list of str, IOB tags for each token
    - token_spans: list of (start, end) tuples or re.Match objects for each token


    Returns:
    - List of chunks, each a list of 'token<TAB>tag' strings
    """

    # Convert Match objects to (start, end) tuples if needed
    token_positions = [(m.start(), m.end()) if hasattr(m, 'start') else m for m in token_spans]


    raw_chunks = text_splitter.split_text(text)

    # Align each chunk manually by finding its first occurrence in text
    used_indices = set()
    chunks = []

    for chunk_text in raw_chunks:
        # Find the first unique match position in text to use as a start index
        start_index = text.find(chunk_text)

        # Prevent collisions if chunk_text repeats (naïve fallback)
        while start_index in used_indices:
            start_index = text.find(chunk_text, start_index + 1)
        used_indices.add(start_index)

        end_index = start_index + len(chunk_text)

        chunk_lines = []
        for (start, end), tag_full in zip(token_positions, iob_data):
            if start_index <= start < end_index:
                token = text[start:end]
                tag = tag_full.split("\t")[1]
                wd = tag_full.split("\t")[0]
                if wd!=token:
                    print("ERROR HERE")
                chunk_lines.append(f"{token}\t{tag}")

        if chunk_lines:
            chunks.append(chunk_lines)

    return chunks



def write_iob_to_file(iob_sentences, output_file_path):
    """
    Writes IOB annotations to a file.

    Args:
        iob_sentences (list of list of str): A list of sentences, where each sentence is a list of strings
                                            in the format "word<TAB>tag".
        output_file_path (str): The path to the output file where the IOB annotations will be written.
    """
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for sentence in iob_sentences:
            for word_tag in sentence:
                file.write(word_tag + '\n')
            # Add an empty line after each sentence
            file.write('\n')


def filter_entities(entities):
    # Sort entities by start index, then by end index (longest first)
    entities.sort(key=lambda x: (x[0], -x[1]))

    # Filter entities to remove internal ranges
    filtered_entities = []
    for entity in entities:
        start, end, entity_name, entity_text = entity
        is_internal = False

        # Check if the current entity is internal to any previously added entity
        for prev_entity in filtered_entities:
            prev_start, prev_end, _, _ = prev_entity
            if start >= prev_start and end <= prev_end:
                is_internal = True
                break

        # If the entity is not internal, add it to the filtered list
        if not is_internal:
            filtered_entities.append(entity)

    return filtered_entities


def convert_conll_to_datasetdict(train_path, val_path=None, test_path=None, label_list=None, unknown_tag="O"):
    """
    Converts CoNLL files to a HuggingFace DatasetDict with typed features.
    If unknown tags are found, they will be replaced with `unknown_tag`.
    """
    if not label_list:
        raise ValueError("You must provide a label_list to define features properly.")

    features = Features({
        "id": Value("string"),
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=label_list))
    })

    label2id = {label: idx for idx, label in enumerate(label_list)}

    def encode_tags(example):
        corrected_tags = []
        for tag in example["ner_tags"]:
            if tag not in label2id:
                print(f"Warning: unknown tag '{tag}' found. Replacing with '{unknown_tag}'")
                tag = unknown_tag
            corrected_tags.append(label2id[tag])
        example["ner_tags"] = corrected_tags
        return example

    data_dict = {}
    for split, path in zip(["train", "validation", "test"], [train_path, val_path, test_path]):
        if path:
            examples = parse_conll_file(path)
            dataset = Dataset.from_list(examples)
            dataset = dataset.map(encode_tags)
            dataset = dataset.cast(features)  # Apply typed features after mapping
            data_dict[split] = dataset

    return DatasetDict(data_dict)


def parse_conll_file(file_path):
    """
    Parses a CoNLL file and returns a list of dicts with 'id', 'tokens', and 'ner_tags'.
    """
    examples = []
    with open(file_path, encoding='utf-8') as f:
        tokens = []
        tags = []
        example_id = 0
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    examples.append({
                        "id": str(example_id),
                        "tokens": tokens,
                        "ner_tags": tags
                    })
                    example_id += 1
                    tokens = []
                    tags = []
            else:
                splits = line.split()
                if len(splits) >= 2:
                    token = splits[0]
                    tag = splits[-1]
                    tokens.append(token)
                    tags.append(tag)
        if tokens:
            examples.append({
                "id": str(example_id),
                "tokens": tokens,
                "ner_tags": tags
            })
    return examples


def ann2iob_singlefile(text_file_path, annotations):
    # Read the text file
    with open(text_file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Parse annotations
    entities = []
    for ann in annotations:
        try:

            entity_name_start_end = int(ann['start_span'])
            entity_name_end_end = int(ann['end_span'])
            entity_text = ann['text']
            entity_type = ann['label']

            entities.append((entity_name_start_end, entity_name_end_end, entity_type, entity_text))
        except Exception as ex:
            print(ann)
            print(ex)
            break

            # Sort entities by start index, and then by length (longer first)
    entities.sort(key=lambda x: (x[0], -x[1]))

    entities = filter_entities(entities)

    # Find the word boundaries within the entity span
    token_spans = split_sentence_with_indices(text)


    iob_data = assign_iob_tags(text, token_spans, entities)
    
    iob_data = chunk_iob_tagged_text(text, iob_data, 
                                     token_spans)
    
    return iob_data


def load_tsv_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Loads a TSV file with specific columns into a pandas DataFrame.

    Expected columns:
        filename, label, start_span, end_span, text, note

    Args:
        file_path (str): Path to the TSV file.

    Returns:
        pd.DataFrame: DataFrame containing the TSV data.
    """
    df = pd.read_csv(
        file_path,
        sep='\t',
        dtype={
            "filename": str,
            "label": str,
            "start_span": int,
            "end_span": int,
            "text": str,
            "note": str
        },
        keep_default_na=False  # Prevents empty strings being converted to NaN
    )
    return df


def generate_iob(txt_root_dict, tsv_file_path, iob_file_path):
    # to save the IOB sentences of all the files
    # load all annotations
    df = load_tsv_to_dataframe(tsv_file_path)

    iob_sentences = []
    for sample_name in tqdm(os.listdir(txt_root_dict)):
        # sample_name = "32277408_ES.txt"
        sample_text_file_path = os.path.join(txt_root_dict, sample_name)

        # get the annotation of this specific sample:
        sample_df = df[df['filename'] == sample_name.replace(".txt", "")]
        if len(sample_df) == 0:
            continue
        sample_annotation = sample_df.to_dict(orient="records")

        # convert the annotation to IOB (all text):
        iob_file = ann2iob_singlefile(text_file_path=sample_text_file_path,
                                          annotations=sample_annotation,
                                         
                                          )
                                
        iob_sentences.extend(iob_file)

    write_iob_to_file(iob_sentences, iob_file_path)
    

cardio_ds_langs = {
    "es":"Spanish",
    "en": "English",
    "cz": "Czech",
    "nl": "Dutch",
    "it": "Italian",
    "ro": "Romanian",
    "sv":"Swedish"
}

for lang_code, lang_name in cardio_ds_langs.items():
    print(lang_code)
    root = f"../dataset/{lang_name}"
    lang = f"{lang_code}"


    label_dict = {
        "dis": ["B-DISEASE", "I-DISEASE", "O"],
        "med": ['B-MEDICATION', 'I-MEDICATION', 'O'],
        "proc": ['B-PROCEDURE', 'I-PROCEDURE', 'O'],
        "symp": ['B-SYMPTOM', 'I-SYMPTOM', 'O'],
    }
    for cat in label_dict.keys():
        print(cat)
        label_list = label_dict[cat]  


        # path to all .txt files
        txt_root_dict = os.path.join(root, "txt")

        # path to train/test annotations
        tsv_file_path_train = os.path.join(root, f"train_cardioccc_{lang}_{cat}.tsv")
        tsv_file_path_test = os.path.join(root,  f"test_cardioccc_{lang}_{cat}.tsv")

        # path to save the IOB files
        iob_file_path_train = os.path.join(root, f"train_cardioccc_{lang}_{cat}.iob")
        iob_file_path_test = os.path.join(root, f"test_cardioccc_{lang}_{cat}.iob")


        generate_iob(txt_root_dict, tsv_file_path_train, iob_file_path_train)
        generate_iob(txt_root_dict, tsv_file_path_test, iob_file_path_test)


        HF_dataset = convert_conll_to_datasetdict(iob_file_path_train, test_path= iob_file_path_test, label_list=label_list)
        HF_dataset
    # ds.save_to_disk(r"dataset/processed_dataset")