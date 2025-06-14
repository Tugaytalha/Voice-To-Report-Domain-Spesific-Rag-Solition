import argparse

from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM as Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

VERBOSE = False

MODEL_NAME = "qwen3:30b"
reasoning_model = True if "qwen3" in MODEL_NAME else False

PROMPT_TEMPLATE = """
Create a radiology report from the doctor's transcript based on the information given in the following context:

{context}

---

Create a radiology report from the doctor's transcript based on the above context: {question}
""" + "\n/nothink" if reasoning_model else ""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--model-type", type=str, help="Specify If model type is sentence_transformer")
    parser.add_argument("--model-name", type=str,
                        default="jinaai/jina-embeddings-v3",
                        help="HuggingFace or Ollama model name or local path")
    args = parser.parse_args()

    # Initialize embedding function with appropriate settings
    embedding_function = get_embedding_function(
        model_name_or_path=args.model_name,
        model_type=args.model_type
    )

    QueryData.query_rag(args.query_text, embedding_function)


class QueryData:
    @staticmethod
    def generate_with_llm(prompt: str, model: str = "llama3.2:3b"):
        """
        Generate text with the given prompt using the LLM model.

        :param prompt: Prompt text to generate the text
        :param model: LLM model name to use from ollama.Default is llama3.2:3b
        :return: Generated text
        """

        model = Ollama(model=model)
        response_text = model.invoke(prompt)

        if VERBOSE:
            print(response_text)

        return response_text

    @staticmethod
    def merge_duplicates(docs_scores):
        """
        Merge(remove) duplicate documents and average their scores.

        :param docs_scores:
        :return: List of unique documents with their average scores
        """
        from collections import defaultdict
        doc_dict = defaultdict(list)  # Stores scores for each unique id
        doc_objects = {}  # Stores a single doc object for each id

        # Aggregate scores for each unique document ID
        for doc, score in docs_scores:
            doc_id = doc.metadata.get("id", None)
            if doc_id is not None:
                doc_dict[doc_id].append(score)
                doc_objects[doc_id] = doc  # Keep one doc object per id

        # Compute average score and reconstruct the list
        result = [(doc_objects[doc_id], sum(scores) / len(scores)) for doc_id, scores in doc_dict.items()]
        return result

    @staticmethod
    def query_rag(query_text: str, embedding_function, model: str = "llama3.2:3b", augmentation: str = None,
                  multi_query: bool = False):
        """
        Query the RAG system with the given query text and get the response.

        :param query_text: Prompt text given by user
        :param embedding_function:  Embedding function itself to use in the vector database
        :param model: LLM model name to use from ollama.Default is llama3.2:3b
        :param augmentation: "query" or "response" to augment the query or response, None(default) to not augment
        :param multi_query: True to generate multiple queries to search in VectorDB, False(default) to generate single query
        . Useful for ambiguous queries or queries that needs retrieval from various documents.

        :return: Response text and chunks with metadata
        """

        # Prepare the DB.
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        search_text = QueryData.augment_query(query_text, model=model, augmentation=augmentation)

        results = []
        if multi_query:
            # Generate multiple queries.
            queries = QueryData.generate_multi_query(search_text, model=model)

            queries = [query.strip() for query in queries] + [search_text]

            # Search the DB with each query (doesn't include the duplicate documents).
            for query in queries:
                results.extend(db.similarity_search_with_score(query, k=4))

            # Remove duplicates.
            results = QueryData.merge_duplicates(results)
            if VERBOSE:
                print(f"Number of unique documents found: {len(results)}")

        else:
            # Search the DB.
            results = db.similarity_search_with_score(search_text, k=5)

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

        response_text = QueryData.generate_with_llm(prompt, model=model)

        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        if VERBOSE:
            print(formatted_response)

        return response_text, chunks_with_metadata

    @staticmethod
    def augment_query(query: str, augmentation: str, model: str = "llama3.2:3b") -> str:
        """
        Augment the query text with the given augmentation type.
        :param augmentation: "query" or "response" to augment the query or response, None(default) to not augment
        Query augmentation is to expand the query to improve searchability and retrieval. Response augmentation is to
        generate an example answer to the given question for optimizing embedding model systems.
        :param model: LLM model name to use from ollama.Default is llama3.2:3b
        :return: Augmented query text
        """
        if augmentation is None or augmentation.lower() == "none":
            return query
        if augmentation.lower() == "answer":
            prompt = f"""You are a helpful expert financial research assistant. 
            Provide an example answer to the given question, that might be found in a document like an annual report. 
            Write only the answer and do not add any words except the answer.
            Question: {query}"""
        elif augmentation.lower() == "query":
            prompt = f"""Given a user's query related to banking operations, financial services, compliance, or customer 
            transactions, generate an expanded version of the query that includes:
            
            Synonyms (e.g., "loan" → "credit facility" or "mortgage financing") 
            Department-specific terminology (e.g., "KYC" → "Know Your Customer compliance process") 
            Regulatory references (e.g., "AML" → "Anti-Money Laundering regulations as per [jurisdiction]") 
            Contextual keywords (e.g., if a query is about 'fraud detection, ' include 'transaction monitoring,' '
            risk assessment,' and 'unauthorized access') 
            Alternative phrasings that match document language Structured categories where applicable (e.g., linking 
            "home loan" with "mortgage rates," "loan tenure," and "EMI calculations") 
            
            Example Input: "What are the rules for opening a corporate bank account?"
            
            Example Expanded Query Output: "Corporate bank account opening guidelines, business banking KYC requirements, 
            corporate account eligibility criteria, required documentation for corporate accounts, company registration 
            verification for banking, commercial account opening compliance, business entity bank account regulations."
            
            Ensure the expanded query maintains relevance, improves searchability, and aligns with banking documentation 
            terminology. Also use same language as the query. Write only the augmented query and do not add any words 
            except the augmented query. Query: {query}"""
        else:
            raise ValueError(f"Invalid augmentation type: {augmentation}")

        return QueryData.generate_with_llm(prompt, model=model)

    @staticmethod
    def generate_multi_query(query: str, model: str = "llama3.2:3b"):
        prompt = f"""You are an advanced banking knowledge assistant optimizing document retrieval for a 
                Retrieval-Augmented Generation (RAG) system. A user has entered a query related to banking services, 
                financial regulations, loans, compliance, legals, or risk management. However, the query may be unclear, broad, or 
                lacking specificity. Your task is to generate three well-structured queries that cover different but relevant 
                aspects of the original query to improve retrieval.
    
                Instructions:
    
                Rephrase the original query while maintaining the intent but targeting different possible document formulations.
                Expand coverage by incorporating industry-specific terminology, regulatory terms, and relevant variations.
                Disambiguate vague queries by assuming possible user intents and generating queries to address them.
                Structure queries effectively to ensure optimal retrieval performance in a banking document database.
                Example 1:
                User Query: "What are the rules for opening a corporate bank account?"
    
                Generated Multi-Queries:
    
                "Corporate bank account opening process and KYC documentation requirements."
                "Regulatory guidelines for corporate account eligibility and compliance."
                "Business banking onboarding policies and required financial records."
                Example 2:
                User Query: "How does fraud detection work in banking?"
    
                Generated Multi-Queries:
    
                "Bank fraud detection techniques and transaction monitoring systems."
                "AML compliance and risk-based fraud detection methods in financial institutions."
                "Machine learning applications in banking fraud prevention and anomaly detection."
                Make sure each query is distinct, optimized for search, and relevant for retrieving high-quality banking documents.
    
                Ensure the queries maintain relevance, improve searchability, and align with banking documentation terminology.
                Use the same language as the original query. Write only the three augmented queries and do not add any words 
                except the augmented queries. Separate each query with a newline. 
                Provide concise single-topic questions (withouth compounding sentences) that cover various aspects of the topic. 
                Ensure each question is complete and directly related to the original inquiry. 
                List each question on a separate line without numbering.
                Query: {query}"""

        response_text = QueryData.generate_with_llm(prompt, model=model)

        if VERBOSE:
            print(response_text)

        return response_text.split("\n")


if __name__ == "__main__":
    main()
