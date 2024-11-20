from models.train import train_model
from models.inference import run_question_generation_model
from models.model import load_model

def main():
    # Paths
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    pretrained_model = "UBC-NLP/AraT5-base"
    output_dir = "./t5_base"

    # Train model
    model = train_model(pretrained_model, train_path, test_path, output_dir)

    # Load model for inference
    model, tokenizer = load_model(output_dir)

    # Run inference
    context = "احمد لعب بالكرة"
    generated_question = run_question_generation_model(context, model, tokenizer)
    print(generated_question)

if __name__ == "__main__":
    main()
