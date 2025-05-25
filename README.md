# Project Title: A RAG-based Document Question & Answering System

This project implements a system that allows users to ask questions about their documents (PDFs) using natural language, including voice input. The system leverages a Retrieval Augmented Generation (RAG) architecture to generate answers using a Large Language Model (LLM).

## Key Features

- **Speech-to-Text Input**: Ask questions using your voice (supports microphone input and audio file uploads).
- **Document Processing**: Ingests PDF documents to build a knowledge base.
- **Retrieval Augmented Generation (RAG)**: Uses RAG to find relevant information from documents and generate answers.
- **Vector Database**: Leverages ChromaDB for efficient similarity search of document chunks.
- **Embeddings**: Supports configurable embedding models (HuggingFace Sentence Transformers and Ollama models like bge-m3).
- **Large Language Model (LLM)**: Utilizes an Ollama model (e.g., llama3.1:8b) for answer generation.
- **Gradio Web Interface**: Provides a user-friendly interface for interacting with the system (Q&A, database management).
- **Standalone Speech-to-Text App**: Includes a separate interface for audio transcription using Whisper.
- **Language Support**: Offers language detection and specification for speech-to-text.

## Project Structure

- `app.py`: Main Gradio application for the Q&A system and database management.
- `populate_database.py`: Script to process PDF documents and populate the Chroma vector database.
- `query_data.py`: Handles querying the database and generating answers using an LLM.
- `stt_app.py`: Separate Gradio application for speech-to-text transcription using Whisper.
- `get_embedding_function.py`: Provides functions to load different embedding models.
- `run_utils.py`: Utility functions connecting transcription, querying, and database population for the main app.
- `requirements.txt`: Lists project dependencies.
- `data/`: Directory where PDF documents should be placed for processing.
- `chroma/`: Directory where ChromaDB stores the vector database.
- `LICENSE`: Contains the project's license information.
- `README.md`: This file.

## Other Files

- `domain_specific_datasets.xlsx`: This Excel file is present in the repository. Its direct programmatic use within the main application flow (e.g., by `populate_database.py` or `app.py`) is not evident from the provided scripts. It might contain datasets for reference, evaluation, or manual data preparation related to domain-specific information.
- `turkish_datasets.xlsx`: Similar to the above, this Excel file's purpose is not explicitly defined within the main application's automated processes. It likely contains datasets, possibly related to Turkish language resources, for reference, evaluation, or manual data preparation tasks.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install FFmpeg:**
    FFmpeg is required for audio processing by Whisper. Download it from the [official FFmpeg download page](https://ffmpeg.org/download.html) and add it to your system's PATH.

5.  **Set up Ollama (if using Ollama for LLM/embeddings):**
    If you plan to use Ollama for LLM or embedding models, download and install it from [Ollama](https://ollama.com/). After installation, pull the necessary models. For example:
    ```bash
    ollama pull llama3.1:8b
    ollama pull bge-m3
    ```

## Usage

1.  **Prepare Documents:**
    Place your PDF files into the `data/` directory. The system will process these documents to build its knowledge base.

2.  **Populate the Database:**
    Run the `populate_database.py` script to process the documents in the `data/` directory and load them into the Chroma vector database.
    
    *   **Basic command:**
        ```bash
        python populate_database.py
        ```
        This will use the default embedding model (Ollama with a model like "bge-m3", or as configured in `get_embedding_function.py`).

    *   **Key arguments:**
        *   `--reset`: Clears the existing database before populating. Use this if you want to start fresh or have updated your documents.
            ```bash
            python populate_database.py --reset
            ```
        *   `--model-type` and `--model-name`: Specify the embedding model to use.
            *   **Using a HuggingFace Sentence Transformer model:**
                ```bash
                python populate_database.py --model-type sentence-transformer --model-name "your-hf-model-name"
                ```
                (e.g., `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr`)
            *   **Using an Ollama embedding model:** (This is the default behavior if `--model-type` is not specified)
                ```bash
                python populate_database.py --model-name "your-ollama-model-name"
                ```
                (e.g., `bge-m3` or `mxbai-embed-large`. Ensure the model is available via Ollama.)
        
        Refer to the "Configuration" section (to be added) for more details on model selection and other parameters.

3.  **Run the Main Q&A Application:**
    Start the Gradio web interface by running:
    ```bash
    python app.py
    ```
    The application provides two main tabs:
    *   **"Query Documents" Tab:**
        *   Upload an audio file or use your microphone to ask a question.
        *   Optionally, specify the language of your audio input for more accurate transcription (e.g., "en", "tr"). If left blank, the system will attempt to auto-detect the language.
        *   Submit your query to receive an answer generated from your documents, along with the source passages.
    *   **"Database Management" Tab:**
        *   **Populate Database:** Click to process documents from the `data/` folder and add them to the vector store. You can select the embedding model type (Sentence Transformer or Ollama) and specify the model name.
        *   **Reset Database:** Click to clear all data from the vector store.

4.  **Run the Standalone Speech-to-Text Application (Optional):**
    For testing speech transcription separately or transcribing audio files without the Q&A functionality, use the `stt_app.py`:
    ```bash
    python stt_app.py
    ```
    This application provides a simple interface to upload or record audio, select a language (or use auto-detect), and view the transcription results.

## Configuration

This section details how to configure various components of the system.

1.  **Embedding Models:**
    Embedding models are crucial for converting text into numerical representations for similarity search.

    *   **Populating the Database (`populate_database.py`):**
        You can specify the embedding model when running the `populate_database.py` script:
        *   **HuggingFace Sentence Transformers:**
            *   Use `--model-type sentence-transformer` and `--model-name <huggingface_model_name_or_path>`.
            *   Example: `python populate_database.py --model-type sentence-transformer --model-name "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"`
            *   If you run `python populate_database.py --model-type sentence-transformer` (without specifying `--model-name`), it will use the default HuggingFace model name defined in `populate_database.py`'s argument parser, which is `"atasoglu/roberta-small-turkish-clean-uncased-nli-stsb-tr"`.
        *   **Ollama Embeddings:**
            *   If `--model-type` is *not* provided to `populate_database.py`, it defaults to using Ollama embeddings. The script calls `get_embedding_function` with `use_sentence_transformer=False`.
            *   Specify the Ollama model with `--model-name <ollama_model_name>`.
            *   Example: `python populate_database.py --model-name "bge-m3"`
            *   If you run `python populate_database.py` (without `--model-type` or `--model-name`), it will use the default Ollama model specified in `get_embedding_function.py` when `use_sentence_transformer=False`, which is `"bge-m3"`. The `--model-name` argument in `populate_database.py` (defaulting to "atasoglu/roberta-small-turkish-clean-uncased-nli-stsb-tr") is ignored in this specific default case because Ollama is selected.

    *   **Default Behavior of `get_embedding_function.py`:**
        *   If called as `get_embedding_function()` (no arguments, or `use_sentence_transformer=True`), it defaults to HuggingFace Sentence Transformer model `"atasoglu/roberta-small-turkish-clean-uncased-nli-stsb-tr"`.
        *   If called as `get_embedding_function(use_sentence_transformer=False)`, it defaults to Ollama model `"bge-m3"`.

    *   **Querying (`app.py` / `run_utils.py` / `query_data.py`):**
        *   The main application (`app.py`) through `run_utils.py` currently uses a **hardcoded** HuggingFace Sentence Transformer model for querying: `"emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"`. This is set in `run_utils.py` when calling `get_embedding_function`.
        *   The `populate_database` function within `run_utils.py` (used by the "Database Management" tab in `app.py`) defaults to using `"emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"` as a Sentence Transformer but allows selection of other models via the UI.
        *   **Important:** If you populate the database with one embedding model (e.g., an Ollama model) and the querying part of the application uses a different one, the system will likely not perform well. To use a different model for querying in the main app, you will need to modify the `get_embedding_function` call within `run_utils.py` in the `process_query` function to match the model used for populating.
        *   The `query_data.py` script (if run directly) has similar command-line arguments as `populate_database.py` for specifying embedding models, but its default `--model-name` is `"emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"`.

2.  **Large Language Model (LLM) for Generation:**
    *   The LLM used for generating answers is specified in `query_data.py`.
    *   It is currently hardcoded to `Ollama(model="llama3.1:8b")`.
    *   To change the LLM, modify this line in `query_data.py`. You can specify a different Ollama model (e.g., `"mistral"`) or replace it with any other Langchain-compatible LLM provider (e.g., `ChatOpenAI`, `HuggingFacePipeline`). Ensure you have the necessary packages and API keys if you switch providers.

3.  **Speech-to-Text (STT) Model:**
    *   The Whisper STT model is configured in `stt_app.py`.
    *   The `MODEL_NAME` variable at the top of the `stt_app.py` script determines which Whisper model is loaded. Options include `"tiny"`, `"base"`, `"small"`, `"medium"`, `"large"`, `"large-v2"`, `"large-v3"`, etc. (check Whisper documentation for all available models). The current default is `"large-v3-turbo"`.
    *   Larger models are more accurate but slower and require more resources (VRAM/RAM).
    *   The main application (`app.py`) uses the transcription function from `stt_app.py` (via `run_utils.py`), so changes to `MODEL_NAME` in `stt_app.py` will affect both the standalone STT app and the main Q&A app.

4.  **Data Path:**
    *   PDF documents are loaded from the directory specified by the `DATA_PATH` variable in `populate_database.py`.
    *   This variable defaults to `"data"`.
    *   You can change this variable at the top of `populate_database.py` if your documents are located elsewhere.

5.  **ChromaDB Path:**
    *   The Chroma vector database is stored in the directory specified by the `CHROMA_PATH` variable.
    *   This variable defaults to `"chroma"`.
    *   It is defined in both `populate_database.py` (for writing) and `query_data.py` (for reading).
    *   If you change this path, ensure you change it consistently in both files. It's also used by `run_utils.py` indirectly through these scripts.

This detailed configuration information should help users adapt the project to their specific needs and available models.

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for the full terms and conditions.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.
