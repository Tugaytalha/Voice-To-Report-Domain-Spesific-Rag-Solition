from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel
from langchain_core.embeddings import Embeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
# import flash_attn


class HostEmbeddings(BaseModel, Embeddings):
    """
    Embeddings client that proxies through your HTTP endpoint.

    Args:
        host: Base URL of your embedding service.
        model: Model name to pass in the payload.
        encode_kwargs: Additional keyword args for batch/document encoding.
        query_encode_kwargs: Keyword args for single-query encoding.
    """
    def __init__(self, host: str = "http://127.0.0.1:38000/embedding",
                 model: str = "Omerhan/intfloat-fine-tuned-14376-v4", encode_kwargs: Dict[str, Any] = {},
                 query_encode_kwargs: Optional[Dict[str, Any]] = None, **data: Any):

        super().__init__(**data)
        self.host = host
        self.model = model
        self.encode_kwargs = encode_kwargs
        self.query_encode_kwargs = query_encode_kwargs

    def _embed(
        self,
        texts: List[str],
        encode_kwargs: Dict[str, Any],
    ) -> List[List[float]]:
        """
        Internal helper to call your HTTP embedding service.
        """
        embeddings: List[List[float]] = []
        for text in texts:
            payload = {
                "model": self.model,
                "prompt": text,
                **encode_kwargs,
            }
            resp = requests.post(self.host, json=payload)
            resp.raise_for_status()
            data = resp.json()
            embeddings.append(data["embedding"])
        return embeddings

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Compute embeddings for a list of documents.
        """
        return self._embed(texts, self.encode_kwargs)  #

    def embed_query(
        self,
        text: str,
    ) -> List[float]:
        """
        Compute an embedding for a single query string.
        """
        kwargs = self.query_encode_kwargs or self.encode_kwargs
        return self._embed([text], kwargs)[0]  #


def get_embedding_function(model_name_or_path="atasoglu/roberta-small-turkish-clean-uncased-nli-stsb-tr",
                           model_type="sentence_transformer", use_cuda=True, host="127.0.0.1:38000/embedding"):  # "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    """
    Get embedding function either from HuggingFace or local directory
    
    Args:
        model_name_or_path (str): Model name or local path
        model_type (str): Model type (sentence_transformer or ollama)
        use_cuda (bool): Use cuda or not
        host (str): Host endpoint

        Returns:
        embeddings: Embedding function
    """
    if model_name_or_path is None:
        model_name_or_path = "atasoglu/roberta-small-turkish-clean-uncased-nli-stsb-tr"
        print("Model name is not provided.")

    if host is None:
        embeddings = HostEmbeddings(host=host, model=model_name_or_path)
        return embeddings
    else:
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


