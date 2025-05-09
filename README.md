# Sentence Classification with Transformers

This project fine-tunes and evaluates a sentence classification model using Hugging Face's Transformers library. It supports multiple classification tasks with stratified k-fold cross-validation and fine-tuning of pre-trained models.

## Features
- Fine-tuning Hugging Face models for sentence classification.
- Stratified k-fold cross-validation for robust evaluation.
- Metrics computation: Precision, Recall, F1-Score, and Accuracy.
- Support for multiple classification tasks.

## Requirements
Install the required Python packages using:
```bash
pip install -r requirements.txt
```

## Usage
Run the script with the following arguments:
```bash
python train_test_model.py <dataset_name> <output_dir_name> <model_name> <num_train_epochs>
```

- dataset_name: Path to the dataset file (tab-separated).
- output_dir_name: Directory to save the model and tokenizer.
- model_name: Hugging Face model name (e.g., bert-base-uncased).
- num_train_epochs: Number of training epochs.

## Example
```bash
python train_test_model.py data/dataset.tsv output bert-base-uncased 3
```

## Dataset Format
The dataset should be a tab-separated file with the following structure:
```
sentence<TAB>fully_curatable<TAB>partially_curatable<TAB>language_related
```

## Output

- Fine-tuned models and tokenizers are saved in the specified output directory.
- Evaluation metrics are printed for each classification task.

## Pre-trained Models
Pretrained models can be found on [Hugging Face](https://huggingface.co/alliance-genome-account).