import os
import time
from pathlib import Path
import torch
import whisper
from tqdm import tqdm
import gc
import argparse
from markdown_pdf import MarkdownPdf, Section

from query_data import QueryData
from get_embedding_function import get_embedding_function

# --- Configuration ---
STEP_DEFAULT = "report"
TEST_DATA_PATH = "test_data/anadolu"
TRANSCRIPT_PATH = "test_data/transcriptions"
LLM_MODEL = "gemma3"
# "symptoma/medgemma3:27b", "gemma3", "alibayram/medgemma:latest"
RESULTS_PATH = "test_data/results" + "_" + LLM_MODEL.split("/")[-1].split(":")[0]
WHISPER_MODEL = "large-v3-turbo"
# Use the same embedding model as your production environment
EMBEDDING_MODEL = "Omerhan/intfloat-fine-tuned-14376-v4"

# CSS to ensure long lines wrap in code blocks for PDF generation
WRAP_CSS = """
pre, code {
    white-space: pre-wrap;
    white-space: -moz-pre-wrap;
    white-space: -pre-wrap;
    white-space: -o-pre-wrap;
    word-wrap: break-word;
}
"""

# --- Global Variables ---
embedding_function = None

def initialize_embedding_function():
    """Loads and initializes the embedding model."""
    global embedding_function
    if embedding_function is not None:
        print("Embedding function already initialized.")
        return

    print("--- Initializing Embedding Model ---")

    # Get embedding function
    print(f"Loading Embedding model: {EMBEDDING_MODEL}...")
    try:
        embedding_function = get_embedding_function(
            model_name_or_path=EMBEDDING_MODEL,
            model_type="sentence_transformer"
        )
        print("✅ Embedding function loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading embedding function: {e}")
        exit()

    print("--- Embedding Model Initialization Complete ---")

def run_transcription_step():
    """
    Finds audio files, transcribes them using Whisper, and saves the
    transcriptions to text files.
    """
    os.makedirs(TRANSCRIPT_PATH, exist_ok=True)
    audio_files = list(Path(TEST_DATA_PATH).glob('**/*.mp3')) + list(Path(TEST_DATA_PATH).glob('**/*.wav'))

    if not audio_files:
        print(f"⚠️ No audio files (.mp3, .wav) found in {TEST_DATA_PATH}. Exiting.")
        return

    print(f"Found {len(audio_files)} audio files to transcribe.")

    for audio_path in tqdm(audio_files, desc="Transcribing audio files"):
        print(f"\nProcessing: {audio_path.name}")
        use_gpu = torch.cuda.is_available()
        device = 'cuda' if use_gpu else 'cpu'
        stt_model = None
        transcription = ""

        try:
            # 1. Load Whisper Model
            if stt_model is None:
                print(f"   Loading Whisper model ({WHISPER_MODEL}) to {device}...")
                stt_model = whisper.load_model(WHISPER_MODEL, device=device)

            # 2. Transcribe audio
            print("   Transcribing audio...")
            result = stt_model.transcribe(str(audio_path), fp16=use_gpu)
            transcription = result["text"]
            print(f"   Transcription (first 50 chars): {transcription[:50]}...")

            # 3. Save the transcription
            transcript_filename = audio_path.stem + ".txt"
            transcript_filepath = Path(TRANSCRIPT_PATH) / transcript_filename
            with open(transcript_filepath, "w", encoding="utf-8") as f:
                f.write(transcription)
            print(f"   ✅ Saved transcription to {transcript_filepath}")

        except Exception as e:
            print(f"   ❌ Error during transcription for {audio_path.name}: {e}")
    
    # 4. Unload Whisper Model
    if stt_model:
        print("   Unloading Whisper model...")
        del stt_model
        if use_gpu:
            torch.cuda.empty_cache()
        gc.collect()
        print("   ✅ Whisper model unloaded.")

def run_report_generation_step():
    """
    Reads transcription files, generates radiology reports using the RAG system,
    and saves them to markdown files.
    """
    os.makedirs(RESULTS_PATH, exist_ok=True)
    transcript_files = list(Path(TRANSCRIPT_PATH).glob("*.txt"))

    if not transcript_files:
        print(f"⚠️ No transcription files (.txt) found in {TRANSCRIPT_PATH}. Exiting.")
        return

    print(f"Found {len(transcript_files)} transcriptions to process.")

    for transcript_path in tqdm(transcript_files, desc="Generating reports"):
        start_time = time.time()
        print(f"\nProcessing: {transcript_path.name}")

        try:
            transcription = transcript_path.read_text(encoding="utf-8")
            if not transcription.strip():
                print(f"   Skipping empty transcription file: {transcript_path.name}")
                continue

            # 1. Generate Report via RAG or without RAG
            rag = False
            print("   Generating report with LLM...")
            if rag:
                response, _ = QueryData.query_rag(
                    query_text=transcription,
                    embedding_function=embedding_function,
                    model=LLM_MODEL,
                    augmentation="None",
                    multi_query=True
                )
            else:
                response = QueryData.query(
                    query_text=transcription,
                    embedding_function=embedding_function,
                    model=LLM_MODEL
                )
            print("   Report generated successfully.")

            # 2. Build markdown content
            md_content = (
                f"# Radiology Report for {transcript_path.name}\n\n"
                "## Transcription\n"
                f"```\n{transcription}\n```\n\n"
                "## AI-Generated Report\n"
                f"{response}"
            )

            # Save markdown (optional, useful for debugging/evaluation)
            md_filename = transcript_path.stem + ".md"
            md_filepath = Path(RESULTS_PATH) / md_filename
            with open(md_filepath, "w", encoding="utf-8") as f:
                f.write(md_content)

            # Convert markdown to PDF
            pdf_filename = transcript_path.stem + ".pdf"
            pdf_filepath = Path(RESULTS_PATH) / pdf_filename
            try:
                pdf = MarkdownPdf(toc_level=2, optimize=True)
                pdf.add_section(Section(md_content), user_css=WRAP_CSS)
                pdf.save(str(pdf_filepath))
            except Exception as pdf_err:
                print(f"   ⚠️ PDF generation failed for {transcript_path.name}: {pdf_err}")

            elapsed_time = time.time() - start_time
            print(f"   ✅ Saved report to {pdf_filepath} (total time: {elapsed_time:.2f}s)")

        except Exception as e:
            print(f"   ❌ Error generating report for {transcript_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Automated Test Framework for Radiology Report Generation.")
    parser.add_argument(
        "--step",
        choices=['transcribe', 'report', 'all'],
        default=STEP_DEFAULT,
        help="Specify which part of the framework to run: 'transcribe' only, 'report' only, or 'all'."
    )
    args = parser.parse_args()

    if args.step in ['transcribe', 'all']:
        print("--- Running Transcription Step ---")
        run_transcription_step()

    if args.step in ['report', 'all']:
        print("\n--- Running Report Generation Step ---")
        initialize_embedding_function()
        run_report_generation_step()

    print("\n--- Test Framework Execution Complete ---")

if __name__ == "__main__":
    main() 