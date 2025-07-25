import gradio as gr
import os
import time
from pathlib import Path
import numpy as np
import whisper
import torch
import warnings
import subprocess
import gc
from markdown_pdf import MarkdownPdf, Section

from query_data import QueryData
from get_embedding_function import get_embedding_function
from visualization_utils import visualize_query_embeddings
from populate_database import get_all_chunk_embeddings
from run_utils import get_ollama_models, handle_file_upload, populate_db_with_params

# Configuration constants
REPORTS_PATH = "generated_reports"
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

# CSS to ensure long lines wrap in code blocks for PDF generation
WRAP_CSS = """
pre, code {
    white-space: pre-wrap; /* CSS3 */
    white-space: -moz-pre-wrap; /* Mozilla */
    white-space: -pre-wrap; /* Opera 4-6 */
    white-space: -o-pre-wrap; /* Opera 7 */
    word-wrap: break-word; /* IE */
}
"""

# Suppress specific warnings if needed (e.g., FP16 on CPU)
warnings.filterwarnings("ignore", category=UserWarning, message="FP16 is not supported on CPU; using FP32 instead")

# Function to transcribe audio to text, now with model loading/unloading
def transcribe_audio(audio_path, model_name, language=None):
    """
    Loads a Whisper model, transcribes the audio, and then unloads the model.
    """
    if audio_path is None:
        return "Error: No audio provided.", ""

    if not os.path.exists(audio_path):
        time.sleep(0.1) # A small delay to allow file system to catch up
        if not os.path.exists(audio_path):
            return f"Error: Audio file not found at path: {audio_path}", ""

    stt_model = None
    try:
        # Check for GPU availability
        print("Checking for GPU...")
        use_gpu = torch.cuda.is_available()
        device = 'cuda' if use_gpu else 'cpu'
        print(f"Using {device} for computation.")

        # Load Whisper model
        print(f"Loading Whisper model: {model_name}...")
        start_time = time.time()
        stt_model = whisper.load_model(model_name, device=device)
        print(f"Model '{model_name}' loaded successfully in {time.time() - start_time:.2f} seconds.")

        # Transcribe
        options = {"fp16": use_gpu, "language": language if language else None}
        options = {k: v for k, v in options.items() if v is not None and v != ""}
        result = stt_model.transcribe(audio_path, **options)

        return f"Transcription successful. Detected language: {result['language']}", result["text"]

    except Exception as e:
        return f"Error during transcription: {str(e)}", ""
    finally:
        # Unload model and clear memory
        if stt_model is not None:
            print("Unloading Whisper model...")
            del stt_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("Whisper model unloaded.")

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
        query_augmentation: str,
        generate_visualization: bool
) -> tuple[str, gr.Dataframe, str, str, str]:
    if not os.path.exists("chroma"):
        return "Error: Database not found. Please populate the database first.", None, "❌ Database not found", None, None

    start_time = time.time()
    status_msg = "🔍 Processing query..."
    report_filepath = None

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

        # Generate visualization (optional)
        visualization_path = None
        if generate_visualization:
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

        # Calculate sources for display
        sources = ", ".join(set([chunk['source'] for chunk in chunks]))
        elapsed_time = time.time() - start_time
        status_msg = f"✅ Query processed in {elapsed_time:.2f} seconds | Sources: {sources}"
        
        # Create markdown content
        md_content = (
            "# Radiology Report\n\n"
            "## Query/Transcription\n\n"
            f"```\n{question}\n```\n\n"
            "## AI-Generated Report\n\n"
            f"{response}\n\n"
            "## Retrieved Sources\n\n"
            f"`{sources}`\n"
        )

        # Ensure reports directory exists
        os.makedirs(REPORTS_PATH, exist_ok=True)

        # Generate PDF filename and path
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        report_filename = f"report_{timestamp}.pdf"
        report_filepath = os.path.join(REPORTS_PATH, report_filename)

        # Convert markdown to PDF
        try:
            pdf = MarkdownPdf(toc_level=2, optimize=True)
            pdf.add_section(Section(md_content), user_css=WRAP_CSS)
            pdf.save(report_filepath)
        except Exception as pdf_err:
            # Fallback: save markdown if PDF generation fails
            fallback_path = report_filepath.replace(".pdf", ".md")
            with open(fallback_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            report_filepath = fallback_path
            print(f"⚠️ PDF generation failed ({pdf_err}). Markdown saved instead: {fallback_path}")

        return response, gr.Dataframe(
            headers=['Source', 'Content', 'Relevance Score'],
            value=df_data
        ), status_msg, visualization_path, report_filepath
    except Exception as e:
        return f"Error processing query: {str(e)}", None, f"❌ Error: {str(e)}", None, None


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
                download_button = gr.File(label="Download Report", interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Model Settings")
                    llm_dropdown = gr.Dropdown(
                        choices=LLM_MODELS,
                        value="gemma3:latest" if "gemma3:latest" in LLM_MODELS else (LLM_MODELS[0] if LLM_MODELS else None),
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
                        value=True,
                        info="Generate multiple search queries to improve retrieval for complex questions"
                    )

                    query_augmentation_dropdown = gr.Dropdown(
                        choices=QUERY_AUGMENTATION_OPTIONS,
                        value=QUERY_AUGMENTATION_OPTIONS[0],
                        label="Query Augmentation",
                        info="How to augment the query for better search results"
                    )

                    generate_viz_checkbox = gr.Checkbox(
                        label="Generate Visualization",
                        value=False,
                        info="Create query-document embedding visualization (may take extra time)"
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
                query_augmentation_dropdown,
                generate_viz_checkbox
            ],
            outputs=[output, chunks_output, status_display, viz_output, download_button]
        )

        transcribe_button.click(
            fn=transcribe_audio,
            inputs=[audio_input, whisper_model_dropdown, lang_input],
            outputs=[status_display, query_input]
        )

    with gr.Tab("Document Management"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Upload Documents")
                file_upload = gr.File(
                    file_types=[".pdf", ".docx", ".txt", ".csv", ".xlsx"],
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
    demo.launch(debug=True, share=True)
