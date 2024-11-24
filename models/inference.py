def run_question_generation_model(input_text: str, model: str, tokenizer):
    # Prepare the input string for the model by appending a task-specific prefix and ending token.
    input_string = "generate a mcq question: " + input_text + " </s>"
    # Tokenize the input string and move to GPU.
    input_ids = tokenizer.encode(input_string, return_tensors="pt").to('cuda')
    generator_args = {
        "max_length": 256, # Maximum length of the generated sequence.
        "num_beams": 4, # Use beam search with 4 beams to generate diverse outputs.
        "length_penalty": 1.5, # Penalize long sequesnce ,shortr sequence is more preferred.
        "no_repeat_ngram_size": 3, # Prevent repeating trigrams in the output.
        "early_stopping": True, # Stop generating when a complete output is done correctly.
    
    }
    # Generate a quesiton by model
    result = model.generate(input_ids, **generator_args)
    # Convert the output tokens into human readable text
    output = tokenizer.batch_decode(result, skip_special_tokens=True)
    return output
