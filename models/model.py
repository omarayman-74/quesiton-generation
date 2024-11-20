#import torch
#from transformers import T5ForConditionalGeneration, T5TokenizerFast
#
#def load_model(pretrained_model_path):
#    model = T5ForConditionalGeneration.from_pretrained(pretrained_model_path)
#    tokenizer = T5TokenizerFast.from_pretrained(pretrained_model_path)
#    return model, tokenizer
#
#def save_model(model, output_dir):
#    model.save_pretrained(output_dir)
#


import pandas as pd
import torch
from datasets import load_dataset, load_metric, list_metrics
from tqdm import tqdm
from typing import Dict, List, Optional
import dataclasses
from dataclasses import dataclass, field
import logging
import sys
import numpy as np
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EvalPrediction,
    DataCollator,
    Trainer,
    T5TokenizerFast,
    TrainingArguments)
from sklearn.model_selection import train_test_split
import torch
from datasets import load_metric
import os
from transformers import Trainer, TrainingArguments, TrainerCallback,TrainerState
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,TrainerControl
import gc
import shutil
from nltk.translate.bleu_score import sentence_bleu


question_answer_dataset = pd.read_csv('dataset.csv')
train_question_answer_dataset, test_question_answer_dataset = train_test_split(question_answer_dataset, test_size=0.10, random_state=42)
train_question_answer_dataset.to_csv('train.csv', index=False)
test_question_answer_dataset.to_csv('test.csv', index=False)

question_generation_dataset = load_dataset('csv', data_files={'train': "train.csv",
                                              'validation': 'test.csv'})


pretrained_model ="UBC-NLP/AraT5-base"
model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
tokenizer = T5TokenizerFast.from_pretrained(pretrained_model)

max_input_length =  256
max_target_length = 256

def convert_to_features(example_batch):

    input_encodings = tokenizer.batch_encode_plus(example_batch['context'],
                                                  max_length=max_input_length,
                                                  add_special_tokens=True,
                                                  truncation=True,
                                                  pad_to_max_length=True)

    target_encodings = tokenizer.batch_encode_plus(example_batch['questions'],
                                                   max_length=max_target_length,
                                                   add_special_tokens=True,
                                                   truncation=True, pad_to_max_length=True)

    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'decoder_input_ids': target_encodings['input_ids']
        ,'decoder_attention_mask': target_encodings['attention_mask']
    }

    return encodings

def prepare_quesition_format(data):
  data['context'] = "generate a mcq question: " +data['context'] + " </s>"
  return data

question_generation_dataset  = question_generation_dataset.map(prepare_quesition_format)
question_generation_dataset  = question_generation_dataset.map(convert_to_features,  batched=True)

columns_to_remove = ["context", "questions"]
train_question_generation_dataset = question_generation_dataset["train"].remove_columns(columns_to_remove)
validation_question_generation_dataset = question_generation_dataset["validation"]

columns = ['input_ids', 'decoder_input_ids', 'attention_mask', 'decoder_attention_mask']
train_question_generation_dataset.set_format(type='torch', columns=columns)
validation_question_generation_dataset.set_format(type='torch', columns=columns)
torch.save(train_question_generation_dataset, 'train_data.pt')
torch.save(validation_question_generation_dataset, 'valid_data.pt')

train_question_generation_dataset = torch.load('train_data.pt')
validation_question_generation_dataset = torch.load('valid_data.pt')

@dataclass
class QuestionGenerationDataCollator():
  def __call__(self, batch: List) -> Dict[str, torch.Tensor]:

    input_ids = torch.stack([example['input_ids'] for example in batch])
    model_labels = torch.stack([example['decoder_input_ids'] for example in batch])
    model_labels[model_labels[:, :] == 0] = -100
    attention_mask = torch.stack([example['attention_mask'] for example in batch])
    decoder_attention_mask = torch.stack([example['decoder_attention_mask'] for example in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': model_labels,
        'decoder_attention_mask': decoder_attention_mask
    }


gc.collect()

os.environ['WANDB_DISABLED'] = 'true'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Save_epochs(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model_path =f'./t5_base-{int(state.epoch)+2}' 
        trainer.save_model(model_path)
        print(f'Model saved for epoch {int(state.epoch)} at {model_path}')


training_args = TrainingArguments(
    output_dir='./t5_base',
    num_train_epochs=8,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,

    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=2500,
    save_strategy='epoch'  )

trainer = Trainer(
    model=model,
    args=training_args,
    train_question_generation_dataset=train_question_generation_dataset,
    eval_dataset=validation_question_generation_dataset,
    data_collator=QuestionGenerationDataCollator(),
    callbacks=[Save_epochs ()]  )

trainer.train(resume_from_checkpoint=True)

trainer.save_model('QG_model')
tokenizer = T5TokenizerFast.from_pretrained("UBC-NLP/AraT5-base")
question_generation_model = T5ForConditionalGeneration.from_pretrained("/kaggle/working/QG_model")

question_generation_model.cuda()



def run_question_generation_model(input_string, **generator_args):
  generator_args = {
  "max_length": 256,
  "num_beams": 4,
  "length_penalty": 1.5,
  "no_repeat_ngram_size": 3,
  "early_stopping": True,
  }
  input_string = "generate a mcq question: " + input_string + " </s>"
  input_ids = tokenizer.encode(input_string, return_tensors="pt").to('cuda')
  result = question_generation_model.generate(input_ids, **generator_args)
  output = tokenizer.batch_decode(result, skip_special_tokens=True)
  output = [item.split("<sep>") for item in output]
  return output

def compute_bleu2(candidate,reference):
    res = 0
    for it in zip(candidate,reference):
        res += (sentence_bleu(it[1], it[0]))
    return res

def test(model,valid_path='test.csv'):
    generator_args = {
                      "max_length": 256,
                      "num_beams": 4,
                      "length_penalty": 1.5,
                      "no_repeat_ngram_size": 3,
                      "early_stopping": True,
                     }
    dt = pd.read_csv(valid_path)
    question_generation_model.to('cuda')
    result = 0
    for item,ques in tqdm(zip(list(dt['context']),list(dt['questions']))):
        input_ids = tokenizer.encode(item+"</s>", return_tensors="pt").to('cuda')
        result = question_generation_model.generate(input_ids, **generator_args)
        output = tokenizer.batch_decode(result, skip_special_tokens=True)
        output = [item.split("<sep>") for item in output]
        que = ques.split("{sep_token}")[0]

        result += (compute_bleu2(str(output[0]).split(),que.split()))
        print(result)
    print(result)

test(question_generation_model)

text = "احمد لعب بالكرة"
run_question_generation_model(text)
