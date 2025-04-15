import gradio as gr
import whisper

# Load the Whisper model
model = whisper.load_model("base")  # You can replace "base" with other model sizes like "small", "medium", "large"

def speech_to_text(audio):
    # Use the whisper model to transcribe audio to text
    result = model.transcribe(audio)
    return result['text']

# Create a Gradio interface for file input
interface = gr.Interface(
    fn=speech_to_text,
    inputs=gr.Audio(source="upload", type="filepath"),  # Accept file uploads (audio files like .wav, .mp3)
    outputs="text"  # Display the transcribed text
)

interface.launch()
