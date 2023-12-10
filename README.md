# BERT Fine-Tuning for Sentiment Analysis

## Overview
This repository contains a Python script for fine-tuning a BERT (Bidirectional Encoder Representations from Transformers) model for sentiment analysis using the IMDB movie reviews dataset. The script utilizes the Hugging Face `transformers` library to load a pre-trained BERT model and fine-tune it on a real-world dataset.

## Requirements
- Python 3.6+
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets

## Installation
Before running the script, ensure you have installed the necessary Python packages. You can install these packages using pip:
```bash
pip install torch transformers datasets
```

## Dataset

The script uses the IMDB dataset, which is a set of movie reviews for binary sentiment classification. The dataset is automatically downloaded and processed by the script.

## Model

The script fine-tunes the bert-base-uncased model, a pre-trained BERT model provided by the Hugging Face Transformers library.

## Usage
To run the fine-tuning script, execute:
'''bash
python bert_fine_tuning.py
'''
This will start the fine-tuning process on the IMDB dataset. The script will save the fine-tuned model in the current directory.

## Output

The fine-tuned model is saved in a directory named fine_tuned_bert. This model can be used for sentiment analysis tasks or further fine-tuning and research.
