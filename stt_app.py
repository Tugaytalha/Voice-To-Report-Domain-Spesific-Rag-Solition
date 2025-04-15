import gradio as gr
import whisper
import os
import time
import warnings
import torch # Required for checking GPU availability and fp16

# --- Configuration ---
MODEL_NAME = "tiny" # Options: "tiny", "base", "small", "medium", "large"
                    # Larger models are more accurate but slower and require more VRAM/RAM.
                    # Start with "base" or "small".

# --- Check for GPU and Load Model ---
# Load the model once when the script starts for efficiency
print("Checking for GPU...")
USE_GPU = torch.cuda.is_available()
compute_dtype = torch.float16 if USE_GPU else torch.float32

print(f"Using {'GPU' if USE_GPU else 'CPU'} for computation.")
print(f"Loading Whisper model: {MODEL_NAME}...")
start_time = time.time()

# Load the model onto GPU if available
try:
    model = whisper.load_model(MODEL_NAME, device='cuda' if USE_GPU else 'cpu')
    # Note: whisper.load_model automatically detects GPU if available when device=None,
    # but specifying it explicitly can be clearer.
    print(f"Model '{MODEL_NAME}' loaded successfully in {time.time() - start_time:.2f} seconds.")

    # Warm-up the model (optional, can reduce latency of the first transcription)
    # print("Warming up the model...")
    # model.transcribe(whisper.silent_audio(30.0), fp16=USE_GPU) # Use fp16=True only if using GPU
    # print("Model warmed up.")

except Exception as e:
    print(f"Error loading Whisper model: {e}")
    print("Please ensure you have installed whisper and its dependencies correctly.")
    print("If using GPU, ensure PyTorch is installed with CUDA support.")
    exit() # Exit if model loading fails

# Suppress specific warnings if needed (e.g., FP16 on CPU)
warnings.filterwarnings("ignore", category=UserWarning, message="FP16 is not supported on CPU; using FP32 instead")

# --- Transcription Function ---
def transcribe_audio(audio_path, language=None):
    """
    Transcribes the given audio file using the pre-loaded Whisper model.

    Args:
        audio_path (str): The file path to the audio file.
        language (str, optional): Language code (e.g., 'en', 'es').
                                    If None, Whisper detects the language. Defaults to None.

    Returns:
        str: The transcribed text or an error message.
    """
    if audio_path is None:
        return "Error: No audio file provided. Please upload an audio file."
    if not os.path.exists(audio_path):
         return f"Error: Audio file not found at path: {audio_path}"

    print(f"\nTranscribing audio file: {audio_path}...")
    transcription_start_time = time.time()

    try:
        # Perform transcription
        # Set fp16=True only if using GPU, otherwise it will warn/error
        options = {"fp16": USE_GPU, "language": language}
        # Remove None values from options dictionary
        options = {k: v for k, v in options.items() if v is not None}

        result = model.transcribe(audio_path, **options)

        transcription = result["text"]
        detected_language = result["language"]
        processing_time = time.time() - transcription_start_time
        print(f"Transcription complete in {processing_time:.2f} seconds.")
        print(f"Detected language: {detected_language}")

        # Optional: Clean up the temporary file created by Gradio if needed
        # (Gradio usually handles this, but good to be aware)
        # if os.path.exists(audio_path):
        #     try:
        #         os.remove(audio_path)
        #         print(f"Cleaned up temporary file: {audio_path}")
        #     except OSError as e:
        #         print(f"Error cleaning up temporary file {audio_path}: {e}")

        return f"Detected Language: {detected_language}\n\nTranscription:\n{transcription}"

    except Exception as e:
        print(f"Error during transcription: {e}")
        return f"Error during transcription: {str(e)}"

# --- Gradio Interface ---
def create_gradio_interface():
    """Creates and returns the Gradio interface."""
    with gr.Blocks(theme=gr.themes.Soft()) as iface:
        gr.Markdown(
            f"""
            # Whisper Speech-to-Text ({MODEL_NAME} model)
            Upload an audio file (e.g., WAV, MP3, M4A, OGG) and click "Transcribe Audio"
            to get the text transcription using the OpenAI Whisper model.
            *Processing time depends on audio length and server load.*
            *Currently using {'GPU' if USE_GPU else 'CPU'}.*
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["upload"], # Only allow file uploads
                    type="filepath",    # Pass the file path to the function
                    label="Upload Audio File"
                )
                lang_input = gr.Textbox(
                    label="Language Code (Optional)",
                    placeholder="e.g., 'en', 'es', 'fr'. Leave blank to auto-detect.",
                    value="" # Default to empty string (auto-detect)
                )
                transcribe_button = gr.Button("Transcribe Audio", variant="primary")

            with gr.Column(scale=2):
                transcription_output = gr.Textbox(
                    label="Transcription Result",
                    placeholder="Transcription will appear here...",
                    lines=15, # Adjust number of lines visible
                    interactive=False # Output field shouldn't be editable by user
                )

        # Connect the button click event to the transcription function
        transcribe_button.click(
            fn=transcribe_audio,
            inputs=[audio_input, lang_input], # Pass both audio path and language
            outputs=transcription_output,
            api_name="transcribe" # Optional: name for API usage
        )

        gr.Examples(
            examples=[
                # Add paths to local example audio files if you have them
                # ["path/to/your/example.wav", "en"],
                # ["path/to/your/example_es.mp3", "es"],
            ],
            inputs=[audio_input, lang_input],
            outputs=transcription_output,
            fn=transcribe_audio,
            cache_examples=False, # Set to True if examples are static and large
            label="Example Audio Files (if available)"
        )

        gr.Markdown("--- \n *Powered by [Gradio](https://gradio.app) and [OpenAI Whisper](https://github.com/openai/whisper)*")

    return iface

# --- Launch the App ---
if __name__ == "__main__":
    app_interface = create_gradio_interface()
    print("Launching Gradio interface...")
    # share=True creates a public link (useful for sharing temporarily, e.g., via Colab)
    # Be cautious with share=True as it exposes your app publicly.
    # Set server_name="0.0.0.0" to allow access from other devices on your network.
    app_interface.launch(server_name="0.0.0.0", server_port=7860)
    # app_interface.launch(share=True)
    # app_interface.launch() # Launches on 127.0.0.1 (localhost) by default