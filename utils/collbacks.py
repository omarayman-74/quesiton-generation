import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("sys.path:", sys.path)

from transformers import TrainerCallback

class SaveEpochs(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        model_path = f'./t5_base-{int(state.epoch) + 1}'
        kwargs['model'].save_pretrained(model_path)
        print(f'Model saved for epoch {int(state.epoch)} at {model_path}')
