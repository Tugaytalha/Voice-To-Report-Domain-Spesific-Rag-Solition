from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
# import flash_attn


def get_embedding_function(model_name_or_path="atasoglu/roberta-small-turkish-clean-uncased-nli-stsb-tr",
                           model_type="sentence_transformer", use_cuda=True):  # "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    """
    Get embedding function either from HuggingFace or local directory
    
    Args:
        model_name_or_path (str): Model name or local path
        model_type (str): Model type (sentence_transformer or ollama)
        use_cuda (bool): Use cuda or not

        Returns:
        embeddings: Embedding function
    """
    if model_name_or_path is None:
        model_name_or_path = "atasoglu/roberta-small-turkish-clean-uncased-nli-stsb-tr"
        print("Model name is not provided.")

    print(f"Using model: {model_name_or_path}")
    if model_type == "sentence_transformer" and use_cuda:
        import torch
        # Check if the cuda is available
        device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        print("Using device: ", device)

        # Create HuggingFaceEmbeddings instance
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name_or_path,
            encode_kwargs={'normalize_embeddings': True
                           },
            model_kwargs={'trust_remote_code': True, 'device': device}
        )

        if device == "cuda":
            embeddings._client = torch.compile(embeddings._client)
    elif model_type == "sentence_transformer":
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name_or_path,
            encode_kwargs={'normalize_embeddings': True
                           },
            model_kwargs={'trust_remote_code': True}
        )
    elif model_type == "ollama":
        embeddings = OllamaEmbeddings(model="bge-m3")
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name_or_path,
            encode_kwargs={'normalize_embeddings': True
                           },
            model_kwargs={'trust_remote_code': True}
        )
        print("Model type is uncertain. Using HuggingFaceEmbeddings model.: ", model_type)

    return embeddings
