import os
import re
from pathlib import Path
from thefuzz import fuzz
from tqdm import tqdm

# --- Configuration ---
RESULTS_PATH = "test_data/results"
GROUND_TRUTH_PATH = "test_data/tranlations"

def normalize_text(text: str) -> str:
    """
    Normalizes text by making it lowercase and removing punctuation/symbols.
    """
    if not text:
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and symbols
    text = re.sub(r'[^\w\s]', '', text)
    # Replace multiple whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_transcription_from_md(md_path: Path) -> str:
    """
    Parses a markdown file and extracts the text from the 'Transcription' code block.
    """
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Find the content within the transcription code block
        match = re.search(r'## Transcription\n```\n(.*?)\n```', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            print(f"⚠️ Could not find transcription block in {md_path.name}")
            return ""
    except Exception as e:
        print(f"❌ Error reading or parsing {md_path.name}: {e}")
        return ""

def evaluate_transcriptions():
    """
    Compares generated transcriptions against ground truth and calculates a similarity score.
    """
    print("--- Starting Transcription Evaluation ---")

    result_files = list(Path(RESULTS_PATH).glob("*.md"))
    if not result_files:
        print(f"⚠️ No result files (.md) found in {RESULTS_PATH}. Run the test framework first.")
        return

    scores = {}
    
    for result_path in tqdm(result_files, desc="Evaluating transcriptions"):
        # e.g., '999_tts_tr' from '999_tts_tr.md'
        base_name = result_path.stem
        # e.g., '999'
        file_id = base_name.split('_')[0] 
        
        # Construct the path to the ground truth file
        # e.g., '999_translated_tr.txt'
        ground_truth_filename = f"{file_id}_translated_tr.txt"
        ground_truth_path = Path(GROUND_TRUTH_PATH) / ground_truth_filename

        if not ground_truth_path.exists():
            print(f"⚠️ No ground truth file found for {result_path.name} at {ground_truth_path}")
            continue

        # Extract and normalize the AI-generated transcription
        ai_transcription = extract_transcription_from_md(result_path)
        normalized_ai = normalize_text(ai_transcription)

        # Read and normalize the ground truth transcription
        try:
            ground_truth_text = ground_truth_path.read_text(encoding="utf-8")
            normalized_gt = normalize_text(ground_truth_text)
        except Exception as e:
            print(f"❌ Error reading ground truth file {ground_truth_path.name}: {e}")
            continue
            
        if not normalized_ai or not normalized_gt:
            print(f"⚠️ Skipping {result_path.name} due to empty normalized text.")
            continue

        # Calculate similarity score
        # fuzz.ratio calculates the standard Levenshtein distance similarity ratio between two sequences.
        score = fuzz.ratio(normalized_ai, normalized_gt)
        scores[result_path.name] = score

    if not scores:
        print("\n--- No scores were calculated. ---")
        return

    # --- Print Results ---
    print("\n--- Evaluation Results ---")
    total_score = 0
    for name, score in scores.items():
        print(f"📄 {name}: {score:.2f}% similarity")
        total_score += score
    
    average_score = total_score / len(scores)
    print("\n--------------------------")
    print(f"📊 Average Similarity Score: {average_score:.2f}%")
    print("--- Evaluation Complete ---")


if __name__ == "__main__":
    evaluate_transcriptions() 