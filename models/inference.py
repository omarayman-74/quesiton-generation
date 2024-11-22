def run_question_generation_model(input_text: str, model: str, tokenizer):
    # adjust the sentence format    
    input_string = "generate a mcq question: " + input_text + " </s>"
    input_ids = tokenizer.encode(input_string, return_tensors="pt").to('cuda')
    generator_args = {
        "max_length": 256,
        "num_beams": 4,
        "length_penalty": 1.5,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
    }
    #generate a text by model
    result = model.generate(input_ids, **generator_args)
    #convert the output tokens into human readable text
    output = tokenizer.batch_decode(result, skip_special_tokens=True)
    return output
