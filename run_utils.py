from query_data import QueryData
from populate_database import _main as populate_db
import os
import gradio as gr
from stt_app import transcribe_audio
from get_embedding_function import get_embedding_function
from langchain_ollama.llms import OllamaLLM
from shutil import copy2
from pathlib import Path
import requests

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def get_ollama_models():
    """
    Fetches the list of available Ollama models.

    It first tries to use the langchain-ollama package. If that fails, it falls
    back to a direct HTTP request to the Ollama API. If both fail, it returns a
    default list of models.
    """
    # Strategy 1: Using LangChain-Ollama Integration
    try:
        # We instantiate with a dummy model name as it's required by the constructor,
        # but it's not used when listing available models via the internal method.
        client = OllamaLLM(model="dummy")
        # _get_models() is an internal method that fetches all model tags.
        # It returns a list of model names as strings.
        model_list = client.list_models()
        if model_list:
            print("Successfully fetched models via langchain-ollama.")
            return model_list
    except Exception as e:
        print(f"Could not fetch models using langchain-ollama: {e}")
        print("Falling back to direct HTTP query.")

    # Strategy 2: Direct HTTP Query (Fallback)
    try:
        OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        resp = requests.get(f"{OLLAMA_URL}/api/tags")
        resp.raise_for_status()
        data = resp.json()
        model_names = [model["name"] for model in data.get("models", [])]
        if model_names:
            print("Successfully fetched models via direct HTTP query.")
            return model_names
        else:
            print("Direct HTTP query returned no models.")
    except requests.exceptions.RequestException as e:
        print(f"Could not fetch models via direct HTTP query: {e}")

    # Fallback to a default list if all methods fail
    print("Falling back to a default list of models.")
    return [
        "llama3.2:3b",
        "llama3.2:1b",
        "llama3.1:8b",
        "llama3.3",
        "llama3.2-vision",
        "gemma3"
    ]


def evaluate_response(actual_response, expected_response):
    """
    Evaluates the actual response against the expected response using an LLM.
    """
    model = OllamaLLM(model="llama3.2:3b")
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


def populate_db_with_params(reset_db, embedding_model):
    """
    Populates the database with the given model.

    :param reset: reset the database
    :param model_name: Embedding model name to use in populating the database
    :param model_type: Embedding model type (sentence_transformer or ollama)
    :return: Success message
    """
    try:
        populate_db(
            reset=reset_db,
            model_name=embedding_model,
            model_type="sentence-transformer"
        )
        return "✅ Database populated successfully!"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def handle_file_upload(files, data_path="data"):
    if not files:
        return "No files uploaded."

    # Create data directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)

    # Copy uploaded files to the data directory
    file_count = 0
    for file in files:
        try:
            filename = Path(file.name).name
            destination = os.path.join(data_path, filename)
            copy2(file.name, destination)
            file_count += 1
        except Exception as e:
            return f"Error copying file {file.name}: {str(e)}"

    return f"✅ Successfully uploaded {file_count} files to the data directory."
