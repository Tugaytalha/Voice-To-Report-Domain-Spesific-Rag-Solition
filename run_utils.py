from query_data import query_rag
from populate_database import main as populate_db
import os
import gradio as gr
from stt_app import transcribe_audio
from get_embedding_function import get_embedding_function


def process_query(audio_path: str, lang = None) -> tuple[str, gr.Dataframe]:
    if not os.path.exists("chroma"):
        return "Error: Database not found. Please populate the database first.", None

    try:
        question = transcribe_audio(audio_path, lang)
        response, chunks = query_rag(question, get_embedding_function("atasoglu/roberta-small-turkish-clean-uncased-nli-stsb-tr"))

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


def populate_database(reset: bool = False, model_name: str = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
                      model_type: str = "sentence-transformer") -> str:
    try:
        import sys

        sys.argv = ["populate_database.py"]
        if reset:
            sys.argv.append("--reset")
        if model_name:
            sys.argv.extend(["--model-type", model_type, "--model-name", model_name])

        populate_db()
        return "Database populated successfully!"
    except Exception as e:
        return f"Error populating database: {str(e)}"
