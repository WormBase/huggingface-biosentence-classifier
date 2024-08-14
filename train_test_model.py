#!/usr/bin/env python
# coding=utf-8
#
# author: Hans-Michael Muller
#

import os
import argparse
from random import shuffle

import evaluate
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from sklearn.model_selection import StratifiedKFold


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    recall = evaluate.load("recall")
    precision = evaluate.load("precision")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels, average='binary')
    rec = recall.compute(predictions=predictions, references=labels, average='binary')
    prec = precision.compute(predictions=predictions, references=labels, average='binary')
    return {**f1_score, **prec, **rec, **acc}


def load_dataset_from_file(dataset_path, class_column_num):
    # Dataset preparation
    dataset = [line.strip().split("\t") for line in open(dataset_path)][1:]
    shuffle(dataset)
    sentences = [line[0] for line in dataset]
    labels = [int(line[class_column_num]) for line in dataset]
    return sentences, labels


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train a sequence classification model.")
    parser.add_argument("dataset_name", type=str, help="Path to the csv dataset.")
    parser.add_argument("output_dir_name", type=str, help="Directory to save the model and tokenizer.")
    parser.add_argument("model_name", type=str, help="Huggingface model to finetune for classification")
    parser.add_argument("num_train_epochs", type=int, help="Number of training epochs.")
    args = parser.parse_args()

    model_name = args.model_name

    for classification_task, classification_column in [('fully_curatable', 1), ('partially_curatable', 2),
                                                       ('language_related', 3)]:
        sentences, labels = load_dataset_from_file(args.dataset_name, classification_column)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encodings = tokenizer(sentences, truncation=True, padding=True, return_tensors="pt")
        encodings["labels"] = torch.tensor(labels)
        tokenized_dataset = Dataset.from_dict(encodings)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir_name, classification_task),
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=args.num_train_epochs,
            weight_decay=0.01,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="epoch",
        )

        precision_scores = []
        recall_scores = []
        f1_scores = []

        # Stratified K-Fold Cross-Validation
        skf = StratifiedKFold(n_splits=5)

        for train_index, test_index in skf.split(tokenized_dataset["input_ids"], tokenized_dataset["labels"]):
            train_dataset = tokenized_dataset.select(train_index)
            eval_dataset = tokenized_dataset.select(test_index)

            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2,
                                                                       ignore_mismatched_sizes=True)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
            trainer.train()
            metrics = trainer.evaluate()
            precision_scores.append(metrics["eval_precision"])
            recall_scores.append(metrics["eval_recall"])
            f1_scores.append(metrics["eval_f1"])

        # Calculate average metrics
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_f1 = np.mean(f1_scores)

        print(f"Classification task: {classification_task}")
        print(f"Precision scores: {precision_scores}")
        print(f"Recall scores: {recall_scores}")
        print(f"F1 Scores: {f1_scores}")
        print(f"Average Precision: {avg_precision}")
        print(f"Average Recall: {avg_recall}")
        print(f"Average F1 Score: {avg_f1}")

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2,
                                                                   ignore_mismatched_sizes=True)

        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir_name, classification_task),
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=args.num_train_epochs,
            weight_decay=0.01,
            save_strategy="epoch",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        trainer.train()
        trainer.save_model()


if __name__ == "__main__":
    main()
