import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--model-type", type=str, help="Specify If model type is sentence-transformer")
    parser.add_argument("--model-name", type=str, 
                       default="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
                       help="HuggingFace or Ollama model name or local path")
    args = parser.parse_args()
    
    # Initialize embedding function with appropriate settings
    embedding_function = get_embedding_function(
        model_name_or_path=args.model_name,
        use_sentence_transformer=bool(args.model_type)
    )
    
    query_rag(args.query_text, embedding_function)


def query_rag(query_text: str, embedding_function):
    # Prepare the DB.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    # Format chunks with their sources and scores
    chunks_with_metadata = []
    for doc, score in results:
        chunks_with_metadata.append({
            'content': doc.page_content,
            'source': doc.metadata.get('id', 'Unknown'),
            'score': f"{score:.4f}"
        })

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="llama3.1:8b")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    
    return response_text, chunks_with_metadata


if __name__ == "__main__":
    main()
