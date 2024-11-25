from transformers import T5ForConditionalGeneration, T5TokenizerFast

def load_model(pretrained_model_path: str):
    """ Load pretrained model and tokenizer."""
    model = T5ForConditionalGeneration.from_pretrained(pretrained_model_path)
    tokenizer = T5TokenizerFast.from_pretrained(pretrained_model_path)
    return model, tokenizer

def save_model(model, output_dir: str):
    """Save the current state of model to dir."""
    model.save_pretrained(output_dir)

