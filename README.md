# Quesiton-generation
# Overview 
This repository provides a model for generating Arabic mcq questions from given contexts using the AraT5 transformers model,for each provided context generate a question and three options , a fine-tuned version of T5 optimized for the Arabic language.
# Technology
main and improtant libraries used in this project are:
transformers: Hugging Face Transformers library for implementing and fine-tuning the T5-based model (AraT5-base)
pytorch: using a pytorch for get a GPU to retrain model on own large dataset
# Model
this project used arat5 pretrain model which based on the transformers architecture and fune tuned on context and questions dataset 
# Data
to prepare the dataset getting a large anount of articles and used nltk to clean it, for prepare data to fitting this project used a stanza library to apply NER (named entity recognation) on the articles then worked with word2vec concepts to get the options from correct answer .
context contain a "generate a mcq question" to guide the model towards generating MCQs.
Text data is tokenized using the T5TokenizerFast from Hugging Face, ensuring the input and output sequences are properly formatted and padded for training.
# Training 
this model trained on p100 GPU with 8 epochs, batch size 4 and learning rate 1e-4
# Project structure 
project/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── train_data.pt
│   ├── valid_data.pt
├── models/
│   ├── model.py
│   ├── train.py
│   ├── inference.py
├── utils/
│   ├── data_processing.py
│   ├── callbacks.py
├── main.py
├── README.md

