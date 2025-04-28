import gradio as gr
from query_data import query_rag
from populate_database import main as populate_db
import os

def process_query(question: str) -> str:
    if not os.path.exists("chroma"):
        return "Error: Database not found. Please populate the database first."

    try:
        response = query_rag(question)
        return response
    except Exception as e:
        return f"Error processing query: {str(e)}"

def populate_database(reset: bool = False) -> str:
    try:
        import sys
        if reset:
            sys.argv = ["populate_database.py", "--reset"]
        else:
            sys.argv = ["populate_database.py"]
        populate_db()
        return "Database populated successfully!"
    except Exception as e:
        return f"Error populating database: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="Document Q&A System") as demo:
    gr.Markdown("# Document Question & Answer System")

    with gr.Tab("Query Documents"):
        query_input = gr.Textbox(
            label="Enter your question",
            placeholder="How much money does a player start with in Monopoly?"
        )
        query_button = gr.Button("Submit Query")
        output = gr.Textbox(label="Response")
        query_button.click(
            fn=process_query,
            inputs=query_input,
            outputs=output
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
    demo.launch()