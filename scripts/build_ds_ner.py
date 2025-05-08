import argparse
import os
import re

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Value
from dotenv import find_dotenv, load_dotenv
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

MIN_WORDS_IOB_SENT = 128
MAX_WORDS_IOB_SENT = 384

print(load_dotenv(find_dotenv(".env")))


def split_sentence_with_indices(text):
    pattern = r'''
        (?:
            \d+[.,]?\d*\s*[%$€]?
        )                                     # Numbers with optional decimal/currency
        |
        [A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:-[A-Za-z0-9]+)*  # Words with optional hyphens (e.g., anti-TNF)
        |
        [()\[\]{}]                             # Parentheses and brackets
        |
        [^\w\s]                                # Other single punctuation marks
    '''
    return list(re.finditer(pattern, text, flags=re.VERBOSE))


def split_iob_into_sentences(iob_tags, min_words=5, max_words=20):
    """
    Splits IOB tags into sub-sentences using NLTK's sentence tokenizer.
    Ensures that sentences have a length between min_words and max_words.

    Args:
        iob_tags (list of str): A list of strings in the format "word<TAB>tag".
        min_words (int): Minimum number of words per sentence.
        max_words (int): Maximum number of words per sentence.

    Returns:
        list of list of str: A list of sub-sentences, where each sub-sentence is a list of "word<TAB>tag" strings.
    """
    # Combine the words into a single text string
    text = ' '.join([word_tag.split('\t')[0] for word_tag in iob_tags])

    # Use NLTK to tokenize the text into sentences
    raw_sentences = sent_tokenize(text)

    # Adjust sentence lengths based on thresholds
    sentences = []
    buffer = []
    for sentence in raw_sentences:
        words = sentence.split()
        buffer.extend(words)

        if len(buffer) >= min_words:
            sentences.append(' '.join(buffer[:max_words]))
            buffer = buffer[max_words:]

    if buffer:
        if sentences and len(sentences[-1].split()) + len(buffer) <= max_words:
            sentences[-1] += ' ' + ' '.join(buffer)
        else:
            sentences.append(' '.join(buffer))

    # Reconstruct the IOB tags for each sentence
    sub_sentences = []
    word_index = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        current_sentence = []

        while word_index < len(iob_tags):
            word, tag = iob_tags[word_index].split('\t')
            current_sentence.append(f"{word}\t{tag}")
            word_index += 1

            if word in sentence_words:
                sentence_words.remove(word)
                if not sentence_words:
                    break

        sub_sentences.append(current_sentence)

    return sub_sentences


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
    with open(text_file_path, 'r', encoding='utf-8-sig') as file:
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

    # Initialize IOB tags
    iob_tags = ['O'] * len(text)

    # Apply IOB tags
    # print(entities)
    for start, end, entity_name, entity_text in entities:
        if 'anfetamínicos' in entity_text:
            pass
        # Find the word boundaries within the entity span
        entity_words = split_sentence_with_indices(
            text)  # list(re.finditer(r'([0-9A-Za-zÀ-ÖØ-öø-ÿ]+|[^0-9A-Za-zÀ-ÖØ-öø-ÿ])', entity_text)) # list(re.finditer(r'\S+', entity_text)) #
        for i, entity_word in enumerate(entity_words):
            if not entity_word.group().strip():
                continue
            word_start = start + entity_word.start()
            word_end = start + entity_word.end()
            if i == 0:
                iob_tags[word_start:word_end] = ['B-' + entity_name] * len(
                    entity_word.group())  # (word_end - word_start)
            else:
                iob_tags[word_start:word_end] = ['I-' + entity_name] * len(
                    entity_word.group())  # (word_end - word_start)

    # Convert the text and IOB tags into word-level IOB format
    text_words = split_sentence_with_indices(
        text)  # list(re.finditer(r'([0-9A-Za-zÀ-ÖØ-öø-ÿ]+|[^0-9A-Za-zÀ-ÖØ-öø-ÿ])', text)) #re.finditer(r'\w+|[^\w\s]', text)
    iob_output = []
    for text_word in text_words:
        word_start = text_word.start()
        word_end = text_word.end()
        word_text = text[word_start:word_end]
        if not word_text.strip():
            continue

        word_letters_tags = iob_tags[word_start:word_end]
        if len(set(word_letters_tags)) == 1:
            word_tag = iob_tags[word_start]
        else:
            word_tag = list(set(word_letters_tags).difference("O"))[0]
        iob_output.append(f"{word_text}\t{word_tag}")

    return iob_output


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

        sample_text_file_path = os.path.join(txt_root_dict, sample_name)

        # get the annotation of this specific sample:
        sample_df = df[df['filename'] == sample_name.replace(".txt", "")]
        if len(sample_df) == 0:
            continue
        sample_annotation = sample_df.to_dict(orient="records")

        # convert the annotation to IOB (all text):
        iob_all_text = ann2iob_singlefile(text_file_path=sample_text_file_path,
                                          annotations=sample_annotation)
        # split the IOB into sentences:
        iob_sentences_single_file = split_iob_into_sentences(iob_all_text,
                                                             min_words=MIN_WORDS_IOB_SENT,
                                                             max_words=MAX_WORDS_IOB_SENT)

        iob_sentences.extend(iob_sentences_single_file)

        # break
    write_iob_to_file(iob_sentences, iob_file_path)


def main(root):
    cardio_ds_langs = {
        "es": "Spanish",
        "en": "English",
        "cz": "Czech",
        "nl": "Dutch",
        "it": "Italian",
        "ro": "Romanian",
        "sv": "Swedish"
    }

    label_dict = {
        "dis": ["B-DISEASE", "I-DISEASE", "O"],
        "med": ['B-MEDICATION', 'I-MEDICATION', 'O'],
        "proc": ['B-PROCEDURE', 'I-PROCEDURE', 'O'],
        "symp": ['B-SYMPTOM', 'I-SYMPTOM', 'O'],
    }

    for lang_code, lang_name in cardio_ds_langs.items():
        print("Processing language:", lang_name)
        lang_root = f"{root}/{lang_name}"
        lang = f"{lang_code}"

        for cat in label_dict.keys():
            print("Processing category:", cat)
            label_list = label_dict[cat]

            # path to all .txt files
            txt_root_dict = os.path.join(lang_root, "txt")

            # path to train/test annotations
            tsv_file_path_train = os.path.join(lang_root, f"train_cardioccc_{lang}_{cat}.tsv")
            tsv_file_path_test = os.path.join(lang_root, f"test_cardioccc_{lang}_{cat}.tsv")

            # path to save the IOB files
            iob_file_path_train = os.path.join(lang_root, f"train_cardioccc_{lang}_{cat}.iob")
            iob_file_path_test = os.path.join(lang_root, f"test_cardioccc_{lang}_{cat}.iob")

            generate_iob(txt_root_dict, tsv_file_path_train, iob_file_path_train)
            print("IOB Path:", iob_file_path_train)
            generate_iob(txt_root_dict, tsv_file_path_test, iob_file_path_test)
            print("IOB Path:", iob_file_path_test)

            HF_dataset = convert_conll_to_datasetdict(iob_file_path_train,
                                                      test_path=iob_file_path_test,
                                                      label_list=label_list)
            HF_dataset
            # ds.save_to_disk(rf"dataset/processed_dataset_{cat}_{lang}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model with specified configuration.")

    parser.add_argument("--root", "-p", type=str, help="Path to the dataset root directory.")

    args = parser.parse_args()

    root = args.root  # ./dataset

    main(root)
