import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("sys.path:", sys.path)

import pandas as pd
from datasets import load_dataset
from transformers import T5TokenizerFast

def prepare_quesition_format(data):
    data['context'] = "generate a mcq question: " + data['context'] + " </s>"
    return data

def convert_to_features(example_batch, tokenizer, max_input_length=256, max_target_length=256):
    input_encodings = tokenizer.batch_encode_plus(
        example_batch['context'],
        max_length=max_input_length,
        add_special_tokens=True,
        truncation=True,
        pad_to_max_length=True
    )

    target_encodings = tokenizer.batch_encode_plus(
        example_batch['questions'],
        max_length=max_target_length,
        add_special_tokens=True,
        truncation=True,
        pad_to_max_length=True
    )

    return {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'decoder_input_ids': target_encodings['input_ids'],
        'decoder_attention_mask': target_encodings['attention_mask']
    }

def load_and_preprocess_data(train_path, test_path, tokenizer):
    question_generation_dataset = load_dataset('csv', data_files={'train': train_path, 'validation': test_path})
    question_generation_dataset = question_generation_dataset.map(prepare_quesition_format)
    question_generation_dataset = question_generation_dataset.map(lambda x: convert_to_features(x, tokenizer), batched=True)
    return question_generation_dataset
