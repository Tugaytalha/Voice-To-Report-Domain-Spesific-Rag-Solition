import os
import time
from pathlib import Path
import torch
import whisper
from tqdm import tqdm
import gc

from query_data import QueryData
from get_embedding_function import get_embedding_function

# --- Configuration ---
TEST_DATA_PATH = "test_data/ttspeechs"
RESULTS_PATH = "test_data/results"
WHISPER_MODEL = "large-v3-turbo"
LLM_MODEL = "gemma3"
# Use the same embedding model as your production environment
EMBEDDING_MODEL = "Omerhan/intfloat-fine-tuned-14376-v4"

# --- Global Variables ---
embedding_function = None

def initialize_embedding_function():
    """Loads and initializes the embedding model."""
    global embedding_function

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


def run_test_suite():
    """
    Runs the full test suite:
    1. Finds all .mp3 files in the test directory.
    2. For each file, it loads whisper, transcribes, unloads whisper,
       then generates a radiology report.
    3. Saves the report to a markdown file.
    """
    print("\n--- Running Test Suite ---")
    os.makedirs(RESULTS_PATH, exist_ok=True)

    audio_files = list(Path(TEST_DATA_PATH).glob("*.mp3"))

    if not audio_files:
        print(f"⚠️ No .mp3 files found in {TEST_DATA_PATH}. Exiting.")
        return

    print(f"Found {len(audio_files)} audio files to process.")

    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        start_time = time.time()
        print(f"\nProcessing: {audio_path.name}")
        use_gpu = torch.cuda.is_available()
        device = 'cuda' if use_gpu else 'cpu'

        # 1. Load Whisper Model
        print(f"   Loading Whisper model ({WHISPER_MODEL}) to {device}...")
        stt_model = whisper.load_model(WHISPER_MODEL, device=device)

        # 2. Transcribe audio
        transcription = ""
        try:
            print("   Transcribing audio...")
            result = stt_model.transcribe(str(audio_path), fp16=use_gpu)
            transcription = result["text"]
            print(f"   Transcription (first 50 chars): {transcription[:50]}...")
        except Exception as e:
            print(f"   ❌ Error during transcription for {audio_path.name}: {e}")
        finally:
            # 3. Unload Whisper Model
            print("   Unloading Whisper model...")
            del stt_model
            if use_gpu:
                torch.cuda.empty_cache()
            gc.collect()
            print("   ✅ Whisper model unloaded.")

        if not transcription:
            print(f"   Skipping {audio_path.name} due to empty or failed transcription.")
            continue

        # 4. Generate Report via RAG
        try:
            print("   Generating report with LLM...")
            response, _ = QueryData.query_rag(
                query_text=transcription,
                embedding_function=embedding_function,
                model=LLM_MODEL,
                augmentation="None",
                multi_query=True
            )
            print("   Report generated successfully.")

            # 5. Save the report
            report_filename = audio_path.stem + ".md"
            report_filepath = Path(RESULTS_PATH) / report_filename

            with open(report_filepath, "w", encoding="utf-8") as f:
                f.write(f"# Radiology Report for {audio_path.name}\n\n")
                f.write("## Transcription\n")
                f.write(f"```\n{transcription}\n```\n\n")
                f.write("## AI-Generated Report\n")
                f.write(response)

            elapsed_time = time.time() - start_time
            print(f"   ✅ Saved report to {report_filepath} (total time: {elapsed_time:.2f}s)")

        except Exception as e:
            print(f"   ❌ Error generating report for {audio_path.name}: {e}")

    print("\n--- Test Suite Complete ---")


if __name__ == "__main__":
    initialize_embedding_function()
    run_test_suite() 