import gradio as gr
import whisper
import os
import time
import warnings
import torch # Required for checking GPU availability and fp16

# --- Configuration ---
# Available Whisper models
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v3-turbo"]
DEFAULT_MODEL = "large-v3-turbo"

# --- Global variables ---
model = None
USE_GPU = False
compute_dtype = None

# Function to load the Whisper model
def load_whisper_model(model_name):
    global model, USE_GPU, compute_dtype
    
    # Check for GPU availability
    if USE_GPU is False:  # Only check once
        print("Checking for GPU...")
        USE_GPU = torch.cuda.is_available()
        compute_dtype = torch.float16 if USE_GPU else torch.float32
        print(f"Using {'GPU' if USE_GPU else 'CPU'} for computation.")
    
    print(f"Loading Whisper model: {model_name}...")
    start_time = time.time()
    
    try:
        model = whisper.load_model(model_name, device='cuda' if USE_GPU else 'cpu')
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

# --- Transcription Function ---
def transcribe_audio(audio_path, language=None):
    """
    Transcribes the given audio file using the pre-loaded Whisper model.
    using the pre-loaded Whisper model.

    Args:
        audio_path (str): The file path to the audio file (could be uploaded or recorded).
        language (str, optional): Language code (e.g., 'en', 'es').
                                    If None, Whisper detects the language. Defaults to None.

    Returns:
        str: The transcribed text or an error message.
    """
    # Gradio might pass None if no audio is provided
    if audio_path is None:
        return "Error: No audio provided. Please upload a file or record audio."

    print(f"\nReceived audio path: {audio_path}") # Path is temporary for recordings

    # Check if the file exists *just in case*
    # Although Gradio's filepath type usually guarantees it exists briefly
    if not os.path.exists(audio_path):
         # Add a small delay and check again, sometimes needed for mic recordings
         time.sleep(0.1)
         if not os.path.exists(audio_path):
            print(f"!!! ERROR: Audio file not found at path: {audio_path}")
            # Include WinError 2 check explicitly based on previous error
            if "WinError 2" in str(os.last_error()) if hasattr(os, 'last_error') else "":
                 return f"Error: Audio file not found. [WinError 2] - Ensure FFmpeg is installed and in your system PATH. Path: {audio_path}"
            return f"Error: Audio file not found at path: {audio_path}"


    print(f"Transcribing audio from: {audio_path}...")
    transcription_start_time = time.time()

    try:
        # Perform transcription
        # Set fp16=True only if using GPU, otherwise it will warn/error
        options = {"fp16": USE_GPU, "language": language if language else None}
        # Remove None values from options dictionary if language is blank
        options = {k: v for k, v in options.items() if v is not None and v != ""}

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
        # Print more detailed error for debugging
        import traceback
        print(f"Error during transcription for file {audio_path}:")
        print(traceback.format_exc()) # Print the full traceback
        # Check specifically for ffmpeg related errors often indicated by WinError 2
        if "WinError 2" in str(e):
             return f"Error during transcription: {str(e)}\n\n>>> This often means FFmpeg was not found. Ensure FFmpeg is installed and added to your system's PATH environment variable, then restart this application. <<<"
        return f"Error during transcription: {str(e)}"
    finally:
        # Optional: Clean up the temporary file created by Gradio, especially for recordings
        # Though Gradio often handles this, explicit cleanup can sometimes help
        # Add a small delay before trying to delete, especially after mic recording
        time.sleep(0.5)
        if audio_path is not None and os.path.exists(audio_path) and "gradio" in audio_path: # Be cautious, only delete Gradio temp files
             try:
                 os.remove(audio_path)
                 print(f"Cleaned up temporary file: {audio_path}")
             except OSError as e:
                 # It's okay if deletion fails sometimes (e.g., file lock)
                 print(f"Warning: Could not clean up temporary file {audio_path}: {e}")


# --- Gradio Interface ---
def create_gradio_interface():
    """Creates and returns the Gradio interface."""
    with gr.Blocks(theme=gr.themes.Soft()) as iface:
        model_info = gr.Markdown(
            f"""
            # Whisper Speech-to-Text
            Upload an audio file (e.g., WAV, MP3, M4A, OGG) **OR** record audio directly from your microphone.
            Click "Transcribe Audio" to get the text transcription using the OpenAI Whisper model.
            *Processing time depends on audio length and server load.*
            *Currently using {'GPU' if USE_GPU else 'CPU'}.*
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                # Model selection dropdown
                model_dropdown = gr.Dropdown(
                    choices=WHISPER_MODELS,
                    value=DEFAULT_MODEL,
                    label="Whisper Model",
                    info="Larger models are more accurate but slower and require more VRAM/RAM."
                )
                model_status = gr.Markdown("")
                
                # Audio input
                audio_input = gr.Audio(
                    sources=["upload", "microphone"], # Allow both sources
                    type="filepath",                  # Keep filepath type for both
                    label="Upload Audio File OR Record from Microphone" # Updated label
                )
                
                # Language input
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

        # Function to update the model when dropdown changes
        def update_model(model_name):
            result = load_whisper_model(model_name)
            return result
        
        # Connect the dropdown change event to update the model
        model_dropdown.change(
            fn=update_model,
            inputs=[model_dropdown],
            outputs=[model_status],
            api_name="update_model"
        )
        
        # Connect the button click event to the transcription function
        transcribe_button.click(
            fn=transcribe_audio,
            inputs=[audio_input, lang_input],
            outputs=transcription_output,
            api_name="transcribe" # Optional: name for API usage
        )

        gr.Examples(
            examples=[
                # Add paths to local example audio files if you have them
                # ["path/to/your/example.wav", "en"],
                # ["path/to/your/example_es.mp3", "es"],
            ],
            # Note: Examples only work for the 'upload' part of the audio component
            inputs=[audio_input, lang_input],
            outputs=transcription_output,
            fn=transcribe_audio,
            cache_examples=False, # Set to True if examples are static and large
            label="Example Audio Files (for Upload)" # Clarified label
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
    app_interface.launch()
    # app_interface.launch(share=True)
    # app_interface.launch() # Launches on 127.0.0.1 (localhost) by default