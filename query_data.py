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
You are an expert AI assistant specializing in generating medical radiology reports. Your task is to convert a raw, unstructured doctor's dictation transcript into a formal, well-structured radiology report in Turkish.

**Instructions:**
1.  Analyze the provided `{question}`.
2.  Extract all relevant medical findings, technical details, and patient metadata.
3.  Structure the information into the following formal sections:
    *   **Header:** Extract `ACC No`, `Islem No`, `Istem Tarihi`, `Çekim Tarihi`, `Onay Tarihi`. If any are missing, leave them blank.
    *   **TETKİK ADI:** The name of the examination.
    *   **TEKNİK:** Describe the imaging technique, parameters, and contrast usage.
    *   **BULGULAR (FINDINGS):** Detail the objective findings in a systematic order (e.g., posterior fossa, supratentorial, ventricles, bones). This should be a descriptive paragraph.
    *   **SONUÇ / YORUM (IMPRESSION / CONCLUSION):** Summarize the most critical findings and provide any recommendations. This should be a concise, numbered list.
4.  The final report must be entirely in **Turkish**.
5.  Maintain a professional, objective, and clinical tone. Do not add any information not present in the transcript.

---
**Context from Knowledge Base:**
{context}

---
**Doctor's Dictation Transcript:**
{question}

---
**Generated Radiology Report:**
""" + "\n/nothink" if reasoning_model else ""


PROMPT_TEMPLATE_WITHOUT_RAG = """
You are an expert AI assistant specializing in generating medical radiology reports. Your task is to convert a raw, unstructured doctor's dictation transcript into a formal, well-structured radiology report in Turkish.

**Instructions:**
1.  Analyze the provided `{question}`.
2.  Extract all relevant medical findings, technical details, and patient metadata.
3.  Structure the information into the following formal sections:
    *   **Header:** Extract `ACC No`, `Islem No`, `Istem Tarihi`, `Çekim Tarihi`, `Onay Tarihi`. If any are missing, leave them blank.
    *   **TETKİK ADI:** The name of the examination.
    *   **TEKNİK:** Describe the imaging technique, parameters, and contrast usage.
    *   **BULGULAR (FINDINGS):** Detail the objective findings in a systematic order (e.g., posterior fossa, supratentorial, ventricles, bones). This should be a descriptive paragraph.
    *   **SONUÇ / YORUM (IMPRESSION / CONCLUSION):** Summarize the most critical findings and provide any recommendations. This should be a concise, numbered list.
4.  The final report must be entirely in **Turkish**.
5.  Maintain a professional, objective, and clinical tone. Do not add any information not present in the transcript.

---
**Generated Radiology Report:**
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
    model = None
    
    @staticmethod
    def init_model(model_name: str):
        """Initialize the Ollama model"""
        if QueryData.model is None:
            QueryData.model = Ollama(model=model_name, num_ctx=64000, temperature=0.0, verbose=True)
        
    @staticmethod
    def generate_with_llm(prompt: str, model: str = "gemma3"):
        """
        Generate text with the given prompt using the LLM model.

        :param prompt: Prompt text to generate the text
        :param model: LLM model name to use from ollama.Default is gemma3
        :return: Generated text
        """
        QueryData.init_model(model)
        response_text = QueryData.model.invoke(prompt)

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
    def query_rag(query_text: str, embedding_function, model: str = "gemma3", augmentation: str = None,
                  multi_query: bool = False):
        """
        Query the RAG system with the given query text and get the response.

        :param query_text: Prompt text given by user
        :param embedding_function:  Embedding function itself to use in the vector database
        :param model: LLM model name to use from ollama.Default is gemma3
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
    def query(query_text: str, embedding_function, model: str = "gemma3"):
        """
        Query the system without RAG with the given query text and get the response.
        """
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_WITHOUT_RAG)
        prompt = prompt_template.format(question=query_text)

        response_text = QueryData.generate_with_llm(prompt, model=model)

        return response_text
                                   
    def augment_query(query: str, augmentation: str, model: str = "gemma3") -> str:
        """
        Augment the query text with the given augmentation type, tailored for radiology.
        :param query: The original query from the user (transcription).
        :param augmentation: "query" for expansion, "answer" for hypothetical document generation.
        :param model: LLM model name to use from ollama.
        :return: Augmented query text.
        """
        if not augmentation or augmentation.lower() == "none":
            return query

        if augmentation.lower() == "answer":
            prompt = f'''You are an expert radiologist. Based on the following clinical observation, generate a concise, hypothetical finding that might appear in a radiology report.
        This finding will be used to find similar documents in a knowledge base.
        Write only the finding and nothing else.

        Clinical Observation: "{query}"

        Hypothetical Finding:'''
        elif augmentation.lower() == "query":
            prompt = f'''You are a medical terminology expert. Expand the following clinical query with relevant radiological terms, synonyms, and alternative phrasings to improve information retrieval from a medical knowledge base.
        Include anatomical locations, potential findings, and related imaging modalities.
        Use the same language as the query. Write only the expanded query.

        Original Query: "{query}"

        Expanded Query:'''
        else:
            raise ValueError(f"Invalid augmentation type: {augmentation}")

        return QueryData.generate_with_llm(prompt, model=model)

    @staticmethod
    def generate_multi_query(query: str, model: str = "gemma3"):
        """
        Generate multiple specific queries from a single clinical transcription for a RAG system
        in a radiology context.
        :param query: The original query from the user (transcription).
        :param model: LLM model name to use from ollama.
        :return: A list of generated queries.
        """
        prompt = f'''You are an expert radiologist and AI assistant. A doctor has provided a transcription of their observations while looking at a medical image.
    Your task is to break down this transcription into 2-5 distinct, specific questions that can be used to query a radiology knowledge base.
    These questions should focus on identifying key findings, anatomical locations, and potential differential diagnoses mentioned or implied in the text.

    - Each question should be a single, concise sentence.
    - The questions should cover different aspects of the transcription.
    - Frame the questions to retrieve detailed descriptions, definitions, or comparison points from a knowledge base.
    - Ensure the generated questions are in the same language as the original transcription.
    - List each question on a separate line without any numbering or bullet points.

    Example:
    Original Transcription: "Patient presents with a persistent cough. Chest X-ray shows a small, ill-defined opacity in the right upper lobe, suspicious for an early-stage malignancy. No signs of pleural effusion or consolidation."

    Generated Queries:
    What are the characteristics of an ill-defined opacity in the right upper lobe on a chest X-ray?
    What are the differential diagnoses for a solitary pulmonary nodule?
    What are the typical radiological signs of early-stage lung malignancy?

    Now, generate the queries for the following transcription:
    Original Transcription: "{query}"

    Generated Queries:'''

        response_text = QueryData.generate_with_llm(prompt, model=model)
        queries = [q.strip() for q in response_text.split("\n") if q.strip()]
        if VERBOSE:
            print(f"Generated multi-queries: {queries}")
        return queries


if __name__ == "__main__":
    main()
