from utils.collbacks import SaveEpochs 
from utils.data_processing import load_and_preprocess_data
from models.model import load_model
from transformers import Trainer ,TrainingArguments

    
def train_model(pretrained_model: str, train_path: str, test_path: str, output_dir: str):
    model, tokenizer = load_model(pretrained_model)
    dataset = load_and_preprocess_data(train_path, test_path, tokenizer)
    
    train_dataset = dataset["train"].remove_columns(["context", "questions"]).set_format(type='torch')
    val_dataset = dataset["validation"].set_format(type='torch')

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=8,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_steps=500,
        weight_decay=0.01,
        save_strategy='epoch',
        logging_dir='./logs'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[SaveEpochs()],
    )

    trainer.train()
    return model


