from query_data import QueryData
from populate_database import _main as populate_db, get_all_chunk_embeddings
import os
import gradio as gr
from stt_app import transcribe_audio
from get_embedding_function import get_embedding_function
from langchain_ollama import OllamaLLM as Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def evaluate_response(actual_response, expected_response):
    """
    Evaluates the actual response against the expected response using an LLM.
    """
    model = Ollama(model="llama3.2:3b")
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=actual_response
    )
    evaluation_result = model.invoke(prompt)
    return evaluation_result.strip()


def process_query(audio_path: str, lang=None, multi_query: bool = False, augmentation: str = None) -> tuple[str, gr.Dataframe]:
    if not os.path.exists("chroma"):
        return "Error: Database not found. Please populate the database first.", None

    try:
        question = transcribe_audio(audio_path, lang)
        embedding_function = get_embedding_function("atasoglu/roberta-small-turkish-clean-uncased-nli-stsb-tr")
        response, chunks = QueryData.query_rag(question, embedding_function, multi_query=multi_query, augmentation=augmentation)

        # Create a DataFrame for display
        df_data = [
            [chunk['source'], chunk['content'], chunk['score']]
            for chunk in chunks
        ]

        return response, gr.Dataframe(
            headers=['Source', 'Content', 'Relevance Score'],
            value=df_data
        )
    except Exception as e:
        return f"Error processing query: {str(e)}", None


def populate_database_wrapper(reset: bool = False, model_name: str = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
                              model_type: str = "sentence-transformer") -> str:
    """
    Populates the database with the given model.

    :param reset: reset the database
    :param model_name: Embedding model name to use in populating the database
    :param model_type: Embedding model type (sentence_transformer or ollama)
    :return: Success message
    """
    try:
        print("I am using this embedding in utils:", model_name)
        populate_db(reset=reset, model_name=model_name, model_type=model_type)
        return "Database populated successfully!"
    except Exception as e:
        return f"Error populating database: {str(e)}"
