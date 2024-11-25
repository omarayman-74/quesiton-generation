from typing import Dict
import pandas as pd
from datasets import load_dataset
from transformers import T5TokenizerFast

def prepare_quesition_format(data: Dict[str, str]):
    """Format sentence to fit the model."""
    data['context'] = "generate a mcq question: " + data['context'] + " </s>"
    return data


def convert_to_features(example_batch: Dict[str, list], tokenizer, max_input_length: int, max_target_length: int) :
    """"Tokenize input and oupu of the model."""
    # Tokenize and format the context by truncate , padding and adding a special token to input of model.
    input_encodings = tokenizer.batch_encode_plus(
        example_batch['context'],
        max_length=max_input_length,
        add_special_tokens=True, # Add special tokens to beginning and end of context. 
        truncation=True, # Truncate the inpute which contain words more than max input length.
        pad_to_max_length=True # Padding the shorter sentence to be equal inpu length. 
    )
    # Tokenize the question as an output.
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

def load_and_preprocess_data(train_path: str, test_path: str, tokenizer):
    """Applying functions on whole data set."""
    # Load dataset.
    question_generation_dataset = load_dataset('csv', data_files={'train': train_path, 'validation': test_path})
    # Apply formatting function to context data to make it suitable for model input.
    question_generation_dataset = question_generation_dataset.map(prepare_quesition_format)
    # Tokenize the context and questions using function apply the specified tokenizer and format them for model input/output.
    question_generation_dataset = question_generation_dataset.map(lambda x: convert_to_features(x, tokenizer), batched=True)
    return question_generation_dataset
