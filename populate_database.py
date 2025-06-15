import argparse
import asyncio
import psutil
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from tqdm.asyncio import tqdm


VERBOSE = True
CHROMA_PATH = "chroma"
DATA_PATH = "data"
MODEL_SIZE_MB = 1100  # TODO: Make this dynamic based on the model size

# Try to initialize NVML for GPU memory management
gpu_available = False
use_cuda = False
if use_cuda:
    import torch
    if torch.cuda.is_available():
        nvmlInit()
        gpu_handle = nvmlDeviceGetHandleByIndex(0)  # Select GPU 0
        gpu_available = True
    else:
        print("⚠ No CUDA device detected. Falling back to CPU processing.")


def main():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--model-type", type=str, help="Specify If model type is sentence_transformer")
    parser.add_argument("--model-name", type=str,
                        default="atasoglu/roberta-small-turkish-clean-uncased-nli-stsb-tr",
                        help="HuggingFace or Ollama model name or local path")
    args = parser.parse_args()

    _main(reset=args.reset, model_name=args.model_name, model_type=args.model_type)


def _main(reset, model_name, model_type):
    if reset:
        print("✨ Clearing Database")
        clear_database()

    print("I am using this embedding in populate:", model_name)

    # Initialize embedding function with appropriate settings
    embedding_function = get_embedding_function(
        model_name_or_path=model_name,
        model_type=model_type
    )

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    asyncio.run(aadd_to_chroma(chunks=chunks, embedding_func=embedding_function))


def get_all_chunk_embeddings():
    """
    Get all chunk embeddings from the database.

    Returns:
        A list of chunk IDs and embeddings.
    """

    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH
    )

    # Get all the documents.
    embeddings = db.get(include=["embeddings"])

    return embeddings


def load_documents():
    document_loader = DirectoryLoader(
        DATA_PATH,
        use_multithreading=True,
        show_progress=VERBOSE,
    )
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


async def get_available_memory():
    """Returns the available memory in MB (GPU if available, otherwise CPU RAM)."""
    if gpu_available:
        info = nvmlDeviceGetMemoryInfo(gpu_handle)
        return info.free // (1024 * 1024)  # Convert bytes to MB
    else:
        return psutil.virtual_memory().available // (1024 * 1024)  # Convert bytes to MB


async def aadd_to_chroma(chunks: list[Document], embedding_func):
    """Add chunks to the Chroma database asynchronously."""

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_func)

    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_ids = set(db.get(include=[])["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
    if not new_chunks:
        print("✅ No new documents to add")
        return

    print(f"👉 Adding new documents: {len(new_chunks)}")

    # Get available memory and adjust batch size
    total_memory = await get_available_memory()
    if gpu_available:
        batch_size = min(500, total_memory // MODEL_SIZE_MB)  # Adjust batch size per model size
        parallel_tasks = 2  # Adjust parallel tasks for GPU
    else:
        batch_size = min(500, total_memory // MODEL_SIZE_MB)  # Adjust batch size per model size
        parallel_tasks = max(1, psutil.cpu_count(logical=False) // 2)  # Half of physical CPU cores

    print(f"Estimated available {'GPU' if gpu_available else 'CPU'} memory: {total_memory} MB")
    print(f"Using batch size: {batch_size}, Parallel tasks: {parallel_tasks}")

    new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
    semaphore = asyncio.Semaphore(parallel_tasks)  # Control concurrent tasks

    async def process_batch(start, end):
        """Process a batch of chunks asynchronously with memory control."""
        async with semaphore:
            try:
                await db.aadd_documents(
                    new_chunks[start:end],
                    ids=new_chunk_ids[start:end],
                    show_progress=VERBOSE
                )
            except Exception as e:
                print(f"❌ Error processing batch {start}-{end}: {e}")

    tasks = [
        process_batch(i, min(i + batch_size, len(new_chunks)))
        for i in range(0, len(new_chunks), batch_size)
    ]

    await tqdm.gather(*tasks, desc="Adding documents to Chroma")
    print("✅ All new documents added successfully.")


def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    print("🗑️ Clearing the database")
    global CHROMA_PATH
    # Remove the database with db.delete_collection()
    Chroma(persist_directory=CHROMA_PATH).delete_collection()
    print("✅ Database cleared")


if __name__ == "__main__":
    main()
