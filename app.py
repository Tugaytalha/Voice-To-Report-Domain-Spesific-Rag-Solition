import gradio as gr
import os
import time
from pathlib import Path
import numpy as np
import whisper
import torch
import warnings
import subprocess

from query_data import QueryData
from get_embedding_function import get_embedding_function
from visualization_utils import visualize_query_embeddings
from populate_database import get_all_chunk_embeddings
from run_utils import get_ollama_models, handle_file_upload, populate_db_with_params

# Configuration constants
EMBEDDING_MODELS = [
    "jinaai/jina-embeddings-v3",
    "Omerhan/intfloat-fine-tuned-14376-v4",
    "intfloat/multilingual-e5-large-instruct",
    "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
    "atasoglu/roberta-small-turkish-clean-uncased-nli-stsb-tr",
    "atasoglu/distilbert-base-turkish-cased-nli-stsb-tr"
]

LLM_MODELS = get_ollama_models()

QUERY_AUGMENTATION_OPTIONS = [
    "None",
    "query",
    "answer"
]

DATA_PATH = "data"

# Available Whisper models
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v3-turbo"]
DEFAULT_MODEL = "tiny"

# Global variables
stt_model = None
USE_GPU = False
compute_dtype = None

# Function to load the Whisper model
def load_whisper_model(model_name):
    global stt_model, USE_GPU, compute_dtype
    
    # Check for GPU availability
    if USE_GPU is False:  # Only check once
        print("Checking for GPU...")
        USE_GPU = torch.cuda.is_available()
        compute_dtype = torch.float16 if USE_GPU else torch.float32
        print(f"Using {'GPU' if USE_GPU else 'CPU'} for computation.")
    
    print(f"Loading Whisper model: {model_name}...")
    start_time = time.time()
    
    try:
        stt_model = whisper.load_model(model_name, device='cuda' if USE_GPU else 'cpu')
        print(f"Model '{model_name}' loaded successfully in {time.time() - start_time:.2f} seconds.")
        return f"Model '{model_name}' loaded successfully."
    except Exception as e:
        error_msg = f"Error loading Whisper model: {e}"
        print(error_msg)
        print("Please ensure you have installed whisper and its dependencies correctly.")
        print("If using GPU, ensure PyTorch is installed with CUDA support.")
        return error_msg

# Suppress specific warnings if needed (e.g., FP16 on CPU)
warnings.filterwarnings("ignore", category=UserWarning, message="FP16 is not supported on CPU; using FP32 instead")

# Initialize model at startup
print("Initializing with default model...")
load_whisper_model(DEFAULT_MODEL)

# Function to transcribe audio to text
def transcribe_audio(audio_path, language=None):
    if audio_path is None:
        # Return a tuple of empty strings for both outputs
        return "Error: No audio provided.", ""

    if not os.path.exists(audio_path):
        time.sleep(0.1)
        if not os.path.exists(audio_path):
            # Return a tuple of empty strings for both outputs
            return f"Error: Audio file not found at path: {audio_path}", ""

    try:
        options = {"fp16": USE_GPU, "language": language if language else None}
        options = {k: v for k, v in options.items() if v is not None and v != ""}
        result = stt_model.transcribe(audio_path, **options)
        # Return a tuple: (status, transcribed_text)
        return f"Transcription successful. Detected language: {result['language']}", result["text"]
    except Exception as e:
        # Return a tuple of error message and empty string
        return f"Error during transcription: {str(e)}", ""
    finally:
        if audio_path is not None and os.path.exists(audio_path) and "gradio" in audio_path:
            try:
                os.remove(audio_path)
            except OSError as e:
                print(f"Warning: Could not clean up temporary file {audio_path}: {e}")


def process_query(
        question: str,
        embedding_model: str,
        llm_model: str,
        use_multi_query: bool,
        query_augmentation: str
) -> tuple[str, gr.Dataframe, str, str]:
    if not os.path.exists("chroma"):
        return "Error: Database not found. Please populate the database first.", None, "❌ Database not found", None

    start_time = time.time()
    status_msg = "🔍 Processing query..."

    try:
        # Get the embedding function
        embedding_func = get_embedding_function(
            model_name_or_path=embedding_model,
            model_type="sentence_transformer"
        )

        # Set augmentation to None if "None" is selected
        actual_augmentation = None if query_augmentation == "None" else query_augmentation

        # Query the RAG model
        response, chunks = QueryData.query_rag(
            query_text=question,
            embedding_function=embedding_func,
            model=llm_model,
            augmentation=actual_augmentation,
            multi_query=use_multi_query
        )

        # Create a DataFrame for display
        df_data = [
            [chunk['source'], chunk['content'], chunk['score']]
            for chunk in chunks
        ]

        # Generate visualization
        visualization_path = None
        # try:
        # Get all chunk embeddings from the database
        all_chunk_data = get_all_chunk_embeddings()
        if any(all_chunk_data) and 'embeddings' in all_chunk_data:
            all_chunk_embeddings = np.array(all_chunk_data["embeddings"])

            # Get query embedding
            query_embedding = np.array(embedding_func.embed_query(question))

            # Get embeddings for retrieved chunks
            retrieved_embeddings = np.array([embedding_func.embed_query(chunk['content']) for chunk in chunks])

            # Create visualization
            visualization_path = visualize_query_embeddings(
                question,
                query_embedding,
                all_chunk_embeddings,
                retrieved_embeddings
            )
        # except Exception as viz_error:
        #     print(f"Visualization error (non-critical): {str(viz_error)}")

        # Calculate sources for display
        sources = ", ".join(set([chunk['source'] for chunk in chunks]))
        elapsed_time = time.time() - start_time
        status_msg = f"✅ Query processed in {elapsed_time:.2f} seconds | Sources: {sources}"

        return response, gr.Dataframe(
            headers=['Source', 'Content', 'Relevance Score'],
            value=df_data
        ), status_msg, visualization_path
    except Exception as e:
        return f"Error processing query: {str(e)}", None, f"❌ Error: {str(e)}", None


# Create the Gradio interface with improved styling
with gr.Blocks(title="InsightBridge AI: Radiology Report Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎙️ InsightBridge AI: From Voice to Verbatim Radiology Reports 📝
        
        Speak or type your observations, and let the AI draft a preliminary radiology report based on your knowledge base.
        """
    )

    with gr.Tab("Query Documents"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Radiology Transcription")
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Upload Audio File OR Record from Microphone"
                )
                with gr.Row():
                    lang_input = gr.Textbox(
                        label="Language Code (Optional)",
                        placeholder="e.g., 'en', 'es', 'fr'. Leave blank to auto-detect.",
                        value=""
                    )
                    whisper_model_dropdown = gr.Dropdown(
                        choices=WHISPER_MODELS,
                        value=DEFAULT_MODEL,
                        label="Whisper Model"
                    )
                transcribe_button = gr.Button("Transcribe to Text", variant="secondary")
                
            with gr.Column(scale=2):
                gr.Markdown("### Transcribed Text")
                query_input = gr.Textbox(
                    label="Transciption",
                    info="The transcribed text from your voice will appear here. You can also edit or type your question/transcription manually.",
                    placeholder="e.g., What are the findings in the patient's chest X-ray?",
                    lines=4
                )
                
                status_display = gr.Textbox(label="Status", interactive=False, lines=2)
        query_button = gr.Button("Submit Query", variant="primary", scale=1)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Response")
                output = gr.Textbox(label="AI Response", lines=5)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Model Settings")
                    llm_dropdown = gr.Dropdown(
                        choices=LLM_MODELS,
                        value=LLM_MODELS[0] if LLM_MODELS else None,
                        label="LLM Model",
                        info="Select the large language model for response generation"
                    )

                    embedding_dropdown = gr.Dropdown(
                        choices=EMBEDDING_MODELS,
                        value=EMBEDDING_MODELS[1],
                        label="Embedding Model",
                        info="Select the embedding model for semantic search (have to be same as vectorDB model)"
                    )

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Advanced Options")
                    multi_query_checkbox = gr.Checkbox(
                        label="Use Multi-Query Generation",
                        value=False,
                        info="Generate multiple search queries to improve retrieval for complex questions"
                    )

                    query_augmentation_dropdown = gr.Dropdown(
                        choices=QUERY_AUGMENTATION_OPTIONS,
                        value=QUERY_AUGMENTATION_OPTIONS[0],
                        label="Query Augmentation",
                        info="How to augment the query for better search results"
                    )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Visualization")
                viz_output = gr.Image(
                    label="Query-Document Visualization",
                    show_download_button=True,
                    type="filepath"
                )
                gr.Markdown("### Retrieved Chunks")
                chunks_output = gr.Dataframe(
                    headers=['Source', 'Content', 'Relevance Score'],
                    label="Retrieved Document Chunks",
                    wrap=True,
                    column_widths=[10, 30, 5]
                )

        query_button.click(
            fn=process_query,
            inputs=[
                query_input,
                embedding_dropdown,
                llm_dropdown,
                multi_query_checkbox,
                query_augmentation_dropdown
            ],
            outputs=[output, chunks_output, status_display, viz_output]
        )

        transcribe_button.click(
            fn=transcribe_audio,
            inputs=[audio_input, lang_input],
            outputs=[status_display, query_input]
        )
        
        whisper_model_dropdown.change(
            fn=load_whisper_model,
            inputs=[whisper_model_dropdown],
            outputs=[status_display]
        )

    with gr.Tab("Document Management"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Upload Documents")
                file_upload = gr.File(
                    file_types=["pdf", "docx", "txt", "csv", "xlsx"],
                    file_count="multiple",
                    label="Upload Files"
                )
                upload_button = gr.Button("Upload Files", variant="primary")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)

            with gr.Column():
                gr.Markdown("### Populate Database")
                with gr.Group():
                    reset_db_checkbox = gr.Checkbox(
                        label="Reset Database",
                        value=False,
                        info="If checked, the existing database will be deleted before populating."
                    )
                    populate_embedding_dropdown = gr.Dropdown(
                        choices=EMBEDDING_MODELS,
                        value=EMBEDDING_MODELS[1],
                        label="Embedding Model",
                        info="Select the embedding model for populating the database."
                    )
                    populate_button = gr.Button("Populate Database", variant="primary")
                    populate_status = gr.Textbox(label="Population Status", interactive=False, lines=3)
        
        upload_button.click(
            fn=handle_file_upload,
            inputs=[file_upload],
            outputs=[upload_status]
        )
        
        populate_button.click(
            fn=populate_db_with_params,
            inputs=[reset_db_checkbox, populate_embedding_dropdown],
            outputs=[populate_status]
        )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(debug=True)
