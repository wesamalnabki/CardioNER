import os
import re
import torch
import torch.nn.functional as F

from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from langchain_text_splitters import RecursiveCharacterTextSplitter
import argparse
from dotenv import find_dotenv, load_dotenv

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



def write_annotations_to_file(data, file_path):
    """
    Writes annotation data to a TSV file.

    Parameters:
        data (list of dict): Each dict should have keys:
            'filename', 'ann_id', 'label', 'start_span', 'end_span', 'text'
        file_path (str): Path to the output file
    """
    header = ['filename', 'ann_id', 'label', 'start_span', 'end_span', 'text']

    with open(file_path, 'w', encoding='utf-8') as f:
        # Write the header
        f.write('\t'.join(header) + '\n')
        # Write each row
        for entry in data:
            row = [str(entry[key]) for key in header]
            f.write('\t'.join(row) + '\n')


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


class PredictionNER:
    def __init__(self, model_checkpoint, revision) -> None:
        MAX_LENGTH = 450
        OVERLAPPING_LEN = 10 

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint, revision=revision, is_split_into_words=True, truncation=False
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint, revision=revision
        )
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name='o200k_base',
            separators=["\n\n\n", "\n\n", "\n", ".", ",", " ", ""],
            keep_separator=False,
            chunk_size=MAX_LENGTH,
            chunk_overlap=OVERLAPPING_LEN,
        )

        ner_labels = list(self.model.config.id2label.values())
        self.base_entity_types = sorted(
            set(label[2:] for label in ner_labels if label != "O")
        )


    def split_text_with_indices(self, text):
        offset = 0
        for doc in self.text_splitter.split_text(text):
            # Search for doc within the remaining text
            start_idx = text.find(doc, offset)
            if start_idx == -1:
                continue  # should not happen, but skip just in case
            end_idx = start_idx + len(doc)
            offset = end_idx  # move search window forward
            yield doc, start_idx, end_idx

    def predict_text(self, text: str, confidence_threshold: float = 0.7):
        # 1. Split text into words and punctuation using regex
        text_matches = split_sentence_with_indices(text)  # list(re.finditer(r'([0-9A-Za-zÀ-ÖØ-öø-ÿ]+|[^0-9A-Za-zÀ-ÖØ-öø-ÿ])', text))

        # 2. Strip and filter out empty or whitespace-only tokens
        text_words = [m.group().strip() for m in text_matches if m.group().strip()]

        if not text_words:
            return []  # return early if nothing valid

        # 3. Tokenize with word alignment
        inputs = self.tokenizer(
            text_words, 
            return_tensors="pt", 
            is_split_into_words=True,
            truncation=False
        )
        word_ids = inputs.word_ids()

        # 4. Predict
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)

        predictions = torch.argmax(logits, dim=2)[0]

        # 5. Map predictions back to original stripped words
        results = []
        seen = set()
        non_empty_matches = [m for m in text_matches if m.group().strip()]

        for i, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx in seen:
                continue
            seen.add(word_idx)

            word = text_words[word_idx]
            tag_id = predictions[i].item()
            tag = self.model.config.id2label[tag_id]
            score = probs[0, i, tag_id].item()
            start = non_empty_matches[word_idx].start()
            end = non_empty_matches[word_idx].end()

            # Apply the confidence threshold filter
            if score < confidence_threshold:
                tag = "O"  # Assign "O" tag if confidence is below threshold
                score = 0.0  # Set score to 0 for "O" tag

            results.append({
                'word': word,
                'tag': tag,
                'start': start,
                'end': end,
                'score': score
            })

        return results

    def aggregate_entities(self, tagged_tokens, original_text, confidence_threshold=0.3):
        # Step 1: Preprocess tags based on the two rules
        corrected_tokens = tagged_tokens.copy()

        # Rule 1: Fix "O" between "B-" and "I-" of the same type
        for i in range(1, len(tagged_tokens) - 1):
            prev_tag = tagged_tokens[i - 1]["tag"]
            curr_tag = tagged_tokens[i]["tag"]
            next_tag = tagged_tokens[i + 1]["tag"]

            if (
                curr_tag == "O" and
                prev_tag.startswith("B-") and
                next_tag.startswith("I-")
            ):
                prev_type = prev_tag[2:]
                next_type = next_tag[2:]
                if prev_type == next_type:
                    corrected_tokens[i]["tag"] = "I-" + prev_type

        # Rule 2: Convert isolated or starting I- to B-
        last_tag_type = None
        for i in range(len(corrected_tokens)):
            tag = corrected_tokens[i]["tag"]
            if tag.startswith("I-"):
                tag_type = tag[2:]
                if last_tag_type != tag_type:
                    corrected_tokens[i]["tag"] = "B-" + tag_type
                    last_tag_type = tag_type
                else:
                    last_tag_type = tag_type
            elif tag.startswith("B-"):
                last_tag_type = tag[2:]
            else:
                last_tag_type = None

        # Step 2: Apply original aggregation logic
        entities = []
        current_entity = None

        for item in corrected_tokens:
            tag = item["tag"]
            start = item["start"]
            end = item["end"]
            score = item["score"]

            if tag.startswith("B-"):
                if current_entity:
                    if all(s >= confidence_threshold for s in current_entity["scores"]):
                        current_entity["text"] = original_text[current_entity["start"]:current_entity["end"]]
                        current_entity["score"] = sum(current_entity["scores"]) / len(current_entity["scores"])
                        del current_entity["scores"]
                        entities.append(current_entity)
                    current_entity = None
                tag_type = tag[2:]
                current_entity = {
                    "start": start,
                    "end": end,
                    "tag": tag_type,
                    "scores": [score]
                }

            elif tag.startswith("I-"):
                tag_type = tag[2:]
                if current_entity and current_entity["tag"] == tag_type:
                    current_entity["end"] = end
                    current_entity["scores"].append(score)
                else:
                    current_entity = {
                        "start": start,
                        "end": end,
                        "tag": tag_type,
                        "scores": [score]
                    }

            else:  # "O"
                if current_entity:
                    if all(s >= confidence_threshold for s in current_entity["scores"]):
                        current_entity["text"] = original_text[current_entity["start"]:current_entity["end"]]
                        current_entity["score"] = sum(current_entity["scores"]) / len(current_entity["scores"])
                        del current_entity["scores"]
                        entities.append(current_entity)
                    current_entity = None

        if current_entity:
            if all(s >= confidence_threshold for s in current_entity["scores"]):
                current_entity["text"] = original_text[current_entity["start"]:current_entity["end"]]
                current_entity["score"] = sum(current_entity["scores"]) / len(current_entity["scores"])
                del current_entity["scores"]
                entities.append(current_entity)

        return entities


    def do_prediction(self, text, confidence_threshold=0.6):
        final_prediction = []
        for sub_text, sub_text_start, sub_text_end in self.split_text_with_indices(text):
            tokens = self.predict_text(text=sub_text, confidence_threshold=confidence_threshold)
            predictions = self.aggregate_entities(tokens, sub_text, confidence_threshold=confidence_threshold)


            for pred in predictions:
                pred["start"] += sub_text_start
                pred["end"] += sub_text_start
                final_prediction.append(pred)

        final_prediction_dict = {
            lab: [p for p in final_prediction if p["tag"] == lab]
            for lab in self.base_entity_types
        }
        merged_predictions = []
        for label in self.base_entity_types:
            merged_predictions.extend(final_prediction_dict[label])
        return merged_predictions


def evaluate(model_checkpoint, revision, root_path, lang, cat):

    ner = PredictionNER(model_checkpoint=model_checkpoint, revision=revision)

    # conver the predictions to ann format
    tsv_file_path_test = os.path.join(root_path,  f"test_cardioccc_{lang}_{cat}.tsv")
    test_files_root =  os.path.join(root_path, "txt")

    test_df = load_tsv_to_dataframe(tsv_file_path_test)
    prd_ann = []

    for fn in tqdm(test_df['filename'].unique()):

        with open(os.path.join(test_files_root, fn+".txt"), 'r', encoding='utf-8') as f:
            document_text = f.read()
            prds = ner.do_prediction(document_text, confidence_threshold=0.35)
            for prd in prds:
                prd_ann.append({
                    "filename": fn,
                    "label": prd["tag"],
                    "ann_id": "NA",
                    "start_span": prd["start"],
                    "end_span": prd["end"],
                    "text": prd["text"],
                })

    output_tsv_path = os.path.join(root_path, f"pre_{model_checkpoint.split('/')[1]}_{revision}.tsv")
    write_annotations_to_file(prd_ann, output_tsv_path)
    print(f"output_tsv_path {output_tsv_path}")

    

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="Run model with specified configuration.")
    
    parser.add_argument("--model_checkpoint", "-m", type=str, help="Model checkpoint to use.")
    parser.add_argument("--revision", "-r", type=str, default="main", help="Model revision or version.")
    parser.add_argument("--root", "-p", type=str, help="Path to the dataset root directory.")
    parser.add_argument("--lang", "-l", type=str,
                        help="Language code (e.g., 'es', 'en').")
    parser.add_argument("--cat", "-c",  type=str, help="Category (e.g., 'med' for medication).")
    
    args = parser.parse_args()
    
    model_checkpoint = args.model_checkpoint
    revision = args.revision
    root = args.root
    lang = args.lang
    cat = args.cat

    lang = lang.upper()
    cat = cat.upper()
    
    evaluate(model_checkpoint, revision, root, lang, cat)


    # Now use these variables below as needed
    print(f"Using model: {model_checkpoint} (revision: {revision})")
    print(f"Dataset root: {root} | Language: {lang} | Category: {cat}")
    
