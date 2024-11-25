
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

class SaveEpochs(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """save weights of each epoch to dir."""
        model_path = f'./t5_base-{int(state.epoch) + 1}'
        kwargs['model'].save_pretrained(model_path)
        print(f'Model saved for epoch {int(state.epoch)} at {model_path}')
