import gradio as gr
from run_utils import *


# Create the Gradio interface
with gr.Blocks(title="AlbaraKa Document Q&A System (Call Center)") as demo:
    gr.Markdown("# Document Question & Answer System")

    with gr.Tab("Query Documents"):
        query_input = gr.Textbox(
            label="Enter your question",
            placeholder="How much money does a player start with in Monopoly?"
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
            inputs=query_input,
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