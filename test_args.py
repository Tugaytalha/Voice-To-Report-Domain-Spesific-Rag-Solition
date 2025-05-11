import argparse
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")    
    parser.add_argument("--model-type", type=str, help="Specify If model type is sentence-transformer")
    parser.add_argument("--model-name", type=str, 
                       default="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
                       help="HuggingFace or Ollama model name or local path")
    args = parser.parse_args()

    # Initialize embedding function with appropriate settings
    embedding_function = get_embedding_function(
        model_name_or_path=args.model_name,
        use_sentence_transformer=bool(args.model_type)
    )





if __name__ == "__main__":
    main()
