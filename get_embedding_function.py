from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from sentence_transformers import SentenceTransformer

def get_embedding_function(model_name_or_path="atasoglu/roberta-small-turkish-clean-uncased-nli-stsb-tr", use_sentence_transformer=True): #"emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    """
    Get embedding function either from HuggingFace or local directory
    
    Args:
        model_name_or_path (str): Model name or local path
        use_sentence_transformer (bool): Whether to use sentence transformer

        Returns:
        embeddings: Embedding function
    """
    if model_name_or_path is None:
        model_name_or_path = "atasoglu/roberta-small-turkish-clean-uncased-nli-stsb-tr"
    if use_sentence_transformer:
        # Create HuggingFaceEmbeddings instance
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name_or_path,
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        embeddings = OllamaEmbeddings(model="bge-m3")
    
    
    return embeddings


