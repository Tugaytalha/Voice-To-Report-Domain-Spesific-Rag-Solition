import gradio as gr
from run_utils import *


# Create the Gradio interface
with gr.Blocks(title="Domain Specific RAG") as demo:
    gr.Markdown("# Document Question & Answer System")

    with gr.Tab("Query Documents"):
        audio_input = gr.Audio(
            sources=["upload", "microphone"],  # Allow both sources
            type="filepath",  # Keep filepath type for both
            label="Upload Audio File OR Record from Microphone"  # Updated label
        )
        lang_input = gr.Textbox(
            label="Language Code (Optional)",
            placeholder="e.g., 'en', 'es', 'fr'. Leave blank to auto-detect.",
            value=""  # Default to empty string (auto-detect)
        )
        query_button = gr.Button("Submit Query")
        output = gr.Textbox(label="Response")
        chunks_output = gr.Dataframe(
            headers=['Source', 'Content', 'Relevance Score'],
            label="Retrieved Chunks",
            wrap=True
        )
        query_button.click(
            fn=process_query,
            inputs=[audio_input, lang_input],
            outputs=[output, chunks_output]
        )

    with gr.Tab("Database Management"):
        gr.Markdown("Populate or reset the document database")
        reset_checkbox = gr.Checkbox(label="Reset Database", value=False)
        populate_button = gr.Button("Populate Database")
        status_output = gr.Textbox(label="Status")
        populate_button.click(
            fn=populate_database,
            inputs=reset_checkbox,
            outputs=status_output
        )

if __name__ == "__main__":
    demo.launch(share=True)