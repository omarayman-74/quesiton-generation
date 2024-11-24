# Quesiton-generation
# Overview 
This project focuses on generating multiple-choice questions (MCQs) in Arabic using the AraT5 model. The model has been fine-tuned on a custom dataset to take a context as input and produce a question with corresponding options as output.

The process integrates advanced natural language processing (NLP) techniques, such as named entity recognition (NER) and word embedding models, to generate accurate and diverse options.

# Technology
main and improtant libraries used in this project are:
transformers: Hugging Face Transformers library for implementing and fine-tuning the T5-based model (AraT5-base)
pytorch: using a pytorch for get a GPU to retrain model on own large dataset
stanza: using stanza to prepare dataset for apltying a NER (named entity recognation) to make a question context 
NLTK: For cleaning and preprocessing the text data.
Word2Vec: For generating plausible distractor options based on the correct answer.

# Model
this project used arat5 pretrain model which based on the transformers architecture and fine tuned on context and questions dataset.

# Data
to prepare the dataset getting a large amount of articles and used nltk to clean it, for prepare data to fitting this project used a stanza library to apply NER (named entity recognation) on the articles then worked with word2vec concepts to get the options from correct answer .
context contain a "generate a mcq question" to guide the model towards generating MCQs.
Text data is tokenized using the T5TokenizerFast from Hugging Face, ensuring the input and output sequences are properly formatted and padded for training.
tokenez.

# Training 
this model trained on p100 GPU with 8 epochs, batch size 4 and learning rate 1e-4.

# Inference 
example intput:
"generate a mcq question:احمد لعب بالكرة"
output:
"... لعب بالكرة"
"احمد" 
"محمد"
"ابراهيم"

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

