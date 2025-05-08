# !pip install -qU datasets  evaluate seqeval langchain-text-splitters

import logging
import os
import uuid
from datetime import datetime

import evaluate
import numpy as np
import wandb
from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Value
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import HfApi, ModelCard, ModelCardData
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, TrainingArguments,
                          Trainer, EarlyStoppingCallback)

print(load_dotenv(find_dotenv(".env")))

# ----------------------------
# Logging Setup
# ----------------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------
# Configuration
# ----------------------------
model_name = "DT4H/CardioBERTa.en"
category = "med"
language = "en"
label_list = ['B-MEDICATION', 'I-MEDICATION', 'O']
use_wandb = True
train_path = f"/content/drive/MyDrive/DT4H_NER/train_cardioccc_{language}_{category}.iob"
test_path = f"/content/drive/MyDrive/DT4H_NER/test_cardioccc_{language}_{category}.iob"

wandb_project = f"dt4h_ner_{language.upper()}_{category.upper()}"
save_directory_wandb = "./save_directory_wandb"
output_path = "./trained_model"
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_version = f"{current_time}-{uuid.uuid4().hex[:8]}"
output_dir = os.path.join(output_path, model_version)
HF_model_push_path = f'DT4H-IE/{model_name.split("/")[-1]}_{language.upper()}_{category.upper()}'


# ----------------------------
# Utility Functions
# ----------------------------
def parse_conll_file(file_path):
    logger.info(f"Parsing CoNLL file from {file_path}")
    examples = []
    with open(file_path, encoding='utf-8') as f:
        tokens, tags, example_id = [], [], 0
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    examples.append({"id": str(example_id), "tokens": tokens, "ner_tags": tags})
                    example_id += 1
                    tokens, tags = [], []
            else:
                splits = line.split()
                if len(splits) >= 2:
                    tokens.append(splits[0])
                    tags.append(splits[-1])
        if tokens:
            examples.append({"id": str(example_id), "tokens": tokens, "ner_tags": tags})
    logger.info(f"Parsed {len(examples)} examples from {file_path}")
    return examples


def convert_conll_to_datasetdict(train_path, test_path, label_list=None, val_size=0.1, unknown_tag="O"):
    logger.info("Converting CoNLL data to DatasetDict")
    if not label_list:
        raise ValueError("You must provide a label_list.")

    features = Features({
        "id": Value("string"),
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=label_list))
    })
    label2id = {label: idx for idx, label in enumerate(label_list)}

    def encode_tags(example):
        example["ner_tags"] = [label2id.get(tag, label2id[unknown_tag]) for tag in example["ner_tags"]]
        return example

    train_dataset = Dataset.from_list(parse_conll_file(train_path)).map(encode_tags)
    split = train_dataset.train_test_split(test_size=val_size, seed=42)
    data_dict = {"train": split["train"].cast(features), "validation": split["test"].cast(features)}

    if test_path:
        test_dataset = Dataset.from_list(parse_conll_file(test_path)).map(encode_tags).cast(features)
        data_dict["test"] = test_dataset

    logger.info("Finished conversion to DatasetDict")
    return DatasetDict(data_dict)


def align_labels_with_tokens(labels, word_ids):
    new_labels, current_word = [], None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            new_labels.append(-100 if word_id is None else labels[word_id])
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            new_labels.append(label + 1 if label % 2 == 1 else label)
    return new_labels


def tokenize_and_align_labels(examples):
    tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    tokenized["labels"] = [align_labels_with_tokens(l, tokenized.word_ids(i)) for i, l in
                           enumerate(examples["ner_tags"])]
    return tokenized


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    true_preds = [[label_names[p] for p, l in zip(pred, lab) if l != -100] for pred, lab in zip(predictions, labels)]
    true_labels = [[label_names[l] for l in lab if l != -100] for lab in labels]
    result = metric.compute(predictions=true_preds, references=true_labels)

    metrics = {k: result[f"overall_{k}"] for k in ["accuracy", "precision", "recall", "f1"]}
    for k, v in result.items():
        if isinstance(v, dict):
            metrics.update({f"{k}_{m}": v[m] for m in v})
    return metrics


# ----------------------------
# Main Training Flow
# ----------------------------
def main():
    global tokenizer, label_names, metric

    logger.info("Starting main training flow")

    if not use_wandb:
        wandb.init(mode="disabled")
        logger.info("WandB disabled")
    else:
        wandb.init(project=wandb_project, name=model_version, dir=save_directory_wandb)
        logger.info(f"WandB initialized for project {wandb_project}")

    raw_datasets = convert_conll_to_datasetdict(train_path, test_path, label_list)
    label_names = raw_datasets["train"].features["ner_tags"].feature.names

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    logger.info(f"Tokenizer loaded from {model_name}")

    tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True,
                                          remove_columns=raw_datasets["train"].column_names)
    logger.info("Tokenization and label alignment completed")

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    metric = evaluate.load("seqeval")
    logger.info("Metric loaded")

    id2label = {i: l for i, l in enumerate(label_names)}
    label2id = {l: i for i, l in id2label.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    logger.info("Model loaded")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        warmup_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="wandb" if use_wandb else None,
        run_name=model_version
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(f'final_model_{model_version}')
    logger.info(f"Model saved as final_model_{model_version}")

    if "test" in tokenized_datasets:
        logger.info("Running evaluation on test set...")
        logits, labels, _ = trainer.predict(tokenized_datasets["test"])
        predictions = np.argmax(logits, axis=-1)
        true_preds = [[label_names[p] for p, l in zip(pred, lab) if l != -100] for pred, lab in
                      zip(predictions, labels)]
        true_labels = [[label_names[l] for l in lab if l != -100] for lab in labels]
        eval_result = evaluate.load("seqeval").compute(predictions=true_preds, references=true_labels)
        logger.info(f"Test evaluation result: {eval_result}")

    logger.info("Pushing model to Hugging Face Hub...")
    api = HfApi()
    api.create_repo(repo_id=HF_model_push_path, exist_ok=True)
    api.create_branch(repo_id=HF_model_push_path, branch=model_version, exist_ok=True)

    model.push_to_hub(
        repo_id=HF_model_push_path,
        branch=model_version,
        revision=model_version,
        commit_message=f"Model uploaded for version {model_version}",
        private=True
    )
    tokenizer.push_to_hub(
        repo_id=HF_model_push_path,
        revision=model_version,
        commit_message=f"Tokenizer uploaded for version {model_version}",
        private=True
    )
    logger.info("Model and tokenizer pushed")

    card_data = ModelCardData(
        language=[language],
        license="apache-2.0",
        tags=["token-classification"],
        datasets=[f"CardioCCC_{language.upper()}_{category.upper()}"],
        metrics=["F1"]
    )
    card = ModelCard.from_template(
        card_data=card_data,
        template="default",
        model_id=HF_model_push_path,
        tag=model_version
    )
    card.save("README.md")
    logger.info("Model card saved")

    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=HF_model_push_path,
        revision=model_version,
        repo_type="model"
    )
    logger.info("README.md uploaded to Hugging Face")


if __name__ == "__main__":
    main()
