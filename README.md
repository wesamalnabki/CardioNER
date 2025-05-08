# Named Entity Recognition (NER) Pipeline

This repository contains a pipeline for building, training, and evaluating Named Entity Recognition (NER) models for multiple languages and entity categories.

## 📁 Project Structure

* `scripts/build_ds_ner.py`: Preprocesses raw text and annotation `.tsv` files into sentence-level IOB format files for NER training.
* `scripts/train_ner.py`: Trains a NER model using the Hugging Face `Trainer` API and pushes the model to the Hugging Face Hub.
* `scripts/evaluate_ner.py`: Evaluates a trained NER model and outputs predictions in `.tsv` format.

---

## 🛠️ 1. Dataset Preparation: `build_ds_ner.py`

This script converts raw annotated clinical text in `.tsv` format into IOB-labeled sentence files for each language.

### 📂 Expected Directory Structure

```
dataset/
├── English/
│   └── txt/
│       ├── file1.txt
│       ├── file1.tsv
│       └── ...
├── Spanish/
│   └── txt/
│       ├── file2.txt
│       ├── file2.tsv
│       └── ...
```

### 📄 TSV File Format

Each `.tsv` file must follow this structure:

```
filename	label	start_span	end_span	text	note
casos_clinicos_cardiologia508	DISEASE	83	95	toxic habits	
casos_clinicos_cardiologia508	DISEASE	130	143	dyslipidaemia	
```

### ⚖️ Customization Parameters

Before running the script, modify the following parameters in `build_ds_ner.py`:

* `MIN_WORDS_IOB_SENT = 128`: Minimum word count per IOB sentence
* `MAX_WORDS_IOB_SENT = 384`: Maximum word count per IOB sentence

### 🔎 Command-line Arguments

* `--root`: Root path to the dataset folder
* `--languages`: Comma-separated list of languages to process (e.g., `English,Spanish`)
* `--categories`: Comma-separated list of entity types to extract (e.g., `MEDICATION,DISEASE,PROCEDURE,SYMPTOM`)

### 📅 Example Usage

```bash
python scripts/build_ds_ner.py --root dataset --languages English,Spanish --categories MEDICATION,DISEASE,PROCEDURE,SYMPTOM
```

The script outputs IOB files in the same language-specific directories.

---

## 🎓 2. Model Training: `train_ner.py`

This script trains a transformer-based NER model and uploads it to the Hugging Face Hub.

### ⚖️ Required Edits Before Running

* `base_model_name`: Name of the pre-trained language model (e.g., `bert-base-multilingual-cased`)
* `language`: Target language (e.g., `English`)
* `category`: Target entity type (e.g., `DISEASE`)
* `labels_list`: List of labels (e.g., `['B-DISEASE', 'I-DISEASE', 'O']`)

### ▶️ Example Usage

```bash
python scripts/train_ner.py
```

The trained model is automatically pushed to Hugging Face under a new branch.

---

## ✍️ 3. Model Evaluation: `evaluate_ner.py`

This script evaluates a trained model and generates predictions in `.tsv` format.

### ⚖️ Required Arguments

* `--lm_path`: Path to the language model on Hugging Face (e.g., `your-username/model-name`)
* `--revision`: Model revision/branch name
* `--language`: Language of the evaluation dataset
* `--category`: Target entity type

### ▶️ Example Usage

```bash
python scripts/evaluate_ner.py --lm_path your-username/model-name --revision disease-english --language English --category DISEASE
```

---

## 📈 Output

* **IOB Files**: Used for training and stored under each language's directory.
* **Trained Model**: Pushed to Hugging Face Hub.
* **Evaluation TSV**: Contains model predictions for comparison with gold annotations.

---

## 🚀 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚪 License

MIT License. See `LICENSE` file for details.
