import subprocess
import sys
import os
import glob
import time
import logging
import xml.etree.ElementTree as ET  # Standard library, should not need install_and_import


# --- Function to ensure packages are available ---
def install_and_import(package_name, import_name=None):
    """Installs a package if not already installed, then imports it."""
    if import_name is None:
        import_name = package_name
    try:
        module = __import__(import_name)
        # If import_name contains dots, __import__ returns the top-level package.
        # We need to navigate to the actual module.
        if '.' in import_name:
            for comp in import_name.split('.')[1:]:
                module = getattr(module, comp)
        print(f"{import_name} is already installed and imported.")
        return module
    except ImportError:
        print(f"{import_name} not found. Attempting to install {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"{package_name} installed successfully.")
            module = __import__(import_name)
            if '.' in import_name:
                for comp in import_name.split('.')[1:]:
                    module = getattr(module, comp)
            return module
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package_name}: {e}")
            if package_name in ["google-generativeai", "google-cloud-texttospeech"]:
                raise ImportError(
                    f"Critical package {package_name} could not be installed. Please install it manually.") from e
            return None  # Return None for non-critical packages if preferred
        except ImportError as e_import:
            print(f"Failed to import {import_name} even after attempting install: {e_import}")
            raise  # Re-raise if import still fails


# --- Attempt to import/install required packages ---
try:
    genai = install_and_import("google-generativeai", "google.generativeai")
    google_cloud_texttospeech = install_and_import("google-cloud-texttospeech", "google.cloud.texttospeech")
except ImportError as e:
    print(f"Exiting due to critical package import failure: {e}")
    sys.exit(1)

# --- Configuration ---
BASE_DIR = "./test_data/"
XML_INPUT_DIR = os.path.join(BASE_DIR, "ecgen-radiology/")
EXTRACTED_TEXT_DIR = os.path.join(BASE_DIR, "extracted_texts/")
TRANSLATED_TEXT_DIR = os.path.join(BASE_DIR, "tranlations/")
TTS_OUTPUT_DIR = os.path.join(BASE_DIR, "ttspeechs/")

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_MODELS_USER_REQUESTED = [
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

GEMMA_MODEL_SDK_NAME = "gemma-3-27b-it"  # Ensure this is a valid model name for SDK
ALL_GEMINI_MODELS_TO_TRY = GEMINI_MODELS_USER_REQUESTED + [GEMMA_MODEL_SDK_NAME]

RPM_DELAYS = {
    15: 60.0 / 15.0,
    30: 60.0 / 30.0,
    "default": 4.0
}

TTS_LANGUAGE_CODE = "tr-TR"
TTS_VOICE_NAME = "tr-TR-Standard-A"
TTS_AUDIO_ENCODING = google_cloud_texttospeech.AudioEncoding.MP3

LOG_FILE = os.path.join(BASE_DIR, "processing_log.log")
logger = logging.getLogger()
if logger.hasHandlers():  # Clear existing handlers to avoid duplicate logs if script is re-run in same session
    logger.handlers.clear()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)


# --- Helper Functions ---
def create_directories():
    os.makedirs(XML_INPUT_DIR, exist_ok=True)
    os.makedirs(EXTRACTED_TEXT_DIR, exist_ok=True)
    os.makedirs(TRANSLATED_TEXT_DIR, exist_ok=True)
    os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
    logging.info("Output directories ensured/created.")


def get_model_delay(model_name):
    if "gemma" in model_name.lower():
        return RPM_DELAYS.get(30, RPM_DELAYS["default"])
    return RPM_DELAYS.get(15, RPM_DELAYS["default"])


# --- Step 1: Text Extraction ---
def extract_text_from_xml(xml_file_path):
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        abstract_texts = []

        abstract_element = root.find(".//Abstract")  # More robust search for Abstract tag
        if abstract_element is None:
            abstract_text_elements = root.findall(".//AbstractText")  # Find all AbstractText if Abstract is missing
            if abstract_text_elements:
                logging.warning(f"No <Abstract> tag found in {xml_file_path}. Found <AbstractText> globally.")
            else:
                logging.warning(f"No <Abstract> or <AbstractText> tags found in {xml_file_path}")
                return None
        else:
            abstract_text_elements = abstract_element.findall("AbstractText")

        if not abstract_text_elements:
            logging.warning(f"No AbstractText elements found in {xml_file_path}")
            return None

        for elem in abstract_text_elements:
            label = elem.get("Label", "").strip()
            # Aggregate all text within the AbstractText element, including mixed content
            text_parts = []
            if elem.text:
                text_parts.append(elem.text.strip())
            for child in elem:
                if child.text:  # Text of child
                    text_parts.append(child.text.strip())
                if child.tail:  # Text after child
                    text_parts.append(child.tail.strip())

            text = " ".join(filter(None, text_parts)).strip()

            text = text.replace("x-XXXX", "x")  # Handle specific replacement first
            text = text.replace("XXXX", "x")  # Then handle general replacement

            if label:  # Append even if text is empty, if label exists, to preserve structure
                abstract_texts.append(f"{label.upper()}: {text}")
            elif text:  # Only append if text exists and no label (less likely with your format)
                abstract_texts.append(text)

        if not abstract_texts:
            logging.warning(f"No usable text content or labels extracted from {xml_file_path}")
            return None

        return " ".join(abstract_texts)
    except ET.ParseError as e:
        logging.error(f"Error parsing XML file: {xml_file_path}. Details: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during XML parsing of {xml_file_path}: {e}")
        return None


# --- Step 2: Translation ---
def translate_text_gemini(text_to_translate, models_to_try, attempt_num_overall=0):
    if not GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY not found. Translation cannot proceed.")
        return None

    if not genai:  # Check if genai was successfully imported
        logging.error("Gemini AI module (genai) not available for translation.")
        return None

    genai.configure(api_key=GEMINI_API_KEY)

    translation_prompt_template = (
        "You are an expert medical translator specializing in radiology. "
        "Translate the following English radiology report text accurately and meticulously into Turkish. "
        "Ensure all medical terminology is translated correctly and the nuances of the original text are preserved. "
        "Do not add any commentary, disclaimers, or any text other than the Turkish translation. Use lower case."
        "If the input text is empty or clearly nonsensical for a radiology context, return an empty string. "
        "TEXT TO TRANSLATE:\n\"\"\"\n{text}\n\"\"\""
    )

    generation_config = genai.types.GenerationConfig(temperature=0.2)

    for i, model_name in enumerate(models_to_try):
        delay = get_model_delay(model_name)
        actual_delay = delay + (attempt_num_overall * 5)

        logging.info(
            f"Attempting translation with model: {model_name} (Try {i + 1}/{len(models_to_try)}, Overall retry factor: {attempt_num_overall}). Waiting {actual_delay:.1f}s.")
        time.sleep(actual_delay)

        try:
            model = genai.GenerativeModel(model_name)
            prompt_with_text = translation_prompt_template.format(text=text_to_translate)

            response = model.generate_content(
                prompt_with_text,
                generation_config=generation_config,
                request_options={'timeout': 180}
            )

            if response.candidates and response.candidates[0].content.parts:
                translated_text = response.text.strip()
                if not translated_text or "cannot fulfill" in translated_text.lower() or "unable to translate" in translated_text.lower():
                    logging.warning(f"Model {model_name} returned empty or refusal: '{translated_text[:100]}...'")
                    continue
                logging.info(f"Successfully translated using {model_name}.")
                return translated_text
            else:
                block_reason_msg = "Unknown reason"
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    block_reason_msg = f"Reason: {response.prompt_feedback.block_reason}"
                    if response.prompt_feedback.safety_ratings:
                        block_reason_msg += f", Safety Ratings: {response.prompt_feedback.safety_ratings}"
                logging.warning(f"No valid content in response from {model_name}. {block_reason_msg}")
                continue
        except Exception as e:
            logging.error(f"Error using Gemini model {model_name}: {e}")
            if "API key not valid" in str(e):
                logging.critical("Gemini API key is invalid. Translation halted.")
                return None
            if any(err_msg in str(e).lower() for err_msg in ["429", "rate limit", "quota", "exhausted"]):
                logging.warning(f"Rate limit/quota likely hit for {model_name}.")
                if i == len(models_to_try) - 1:
                    return "RATE_LIMIT_RETRY"

    logging.error(f"All Gemini models failed for translation of text: '{text_to_translate[:70]}...'")
    return None


# --- Step 3: Text-to-Speech ---
def synthesize_speech_gcloud(text_to_synthesize, output_audio_path):
    if not google_cloud_texttospeech:
        logging.error("Google Cloud Text-to-Speech module not available.")
        return False
    try:
        client = google_cloud_texttospeech.TextToSpeechClient()
        input_text = google_cloud_texttospeech.SynthesisInput(text=text_to_synthesize)
        voice = google_cloud_texttospeech.VoiceSelectionParams(
            language_code=TTS_LANGUAGE_CODE,
            name=TTS_VOICE_NAME
        )
        audio_config = google_cloud_texttospeech.AudioConfig(
            audio_encoding=TTS_AUDIO_ENCODING,
            speaking_rate=0.9
        )
        response = client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )
        with open(output_audio_path, "wb") as out_file:
            out_file.write(response.audio_content)
        logging.info(f"Audio content written to file: {output_audio_path}")
        return True
    except Exception as e:
        logging.error(f"Error during TTS for {output_audio_path}: {e}")
        if "Could not find Application Default Credentials" in str(e):
            logging.critical("Google Cloud credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS.")
        return False


# --- Main Processing Logic ---
def process_files_main(max_files_to_process_override=None, start_from_file_index=0):
    if not GEMINI_API_KEY:
        logging.critical("GOOGLE_API_KEY environment variable not set. Exiting.")
        return
    try:
        if google_cloud_texttospeech:
            google_cloud_texttospeech.TextToSpeechClient()
        logging.info("Google Cloud TTS client initialized successfully (credentials likely ok).")
    except Exception as e:
        if "Could not find Application Default Credentials" in str(e) or \
                "missing_default_credentials" in str(e).lower():  # More specific check
            logging.critical(
                "Google Cloud TTS credentials (GOOGLE_APPLICATION_CREDENTIALS) not found or invalid. Exiting.")
            return
        logging.warning(f"Google Cloud TTS client init warning: {e}. Will proceed, but TTS might fail.")

    create_directories()

    all_xml_files = sorted(glob.glob(os.path.join(XML_INPUT_DIR, "*.xml")))
    if not all_xml_files:
        logging.info(f"No XML files found in {XML_INPUT_DIR}.")
        return

    # Apply start_from_file_index
    files_to_process_this_run_segment = all_xml_files[start_from_file_index:]

    logging.info(
        f"Found {len(all_xml_files)} total XML files. Starting from index {start_from_file_index} ({len(files_to_process_this_run_segment)} files in this batch segment).")

    # Apply max_files_to_process_override to the current segment
    if max_files_to_process_override is not None:
        files_to_process_this_run_final = files_to_process_this_run_segment[:max_files_to_process_override]
        logging.info(
            f"Processing a maximum of {len(files_to_process_this_run_final)} files in this specific run due to override.")
    else:
        files_to_process_this_run_final = files_to_process_this_run_segment
        logging.info(
            f"Processing all {len(files_to_process_this_run_final)} files from start index {start_from_file_index}.")

    newly_processed_stages_count = 0
    overall_translation_attempt_num = 0

    for current_xml_absolute_idx, xml_file_path in enumerate(files_to_process_this_run_final,
                                                             start=start_from_file_index):
        base_filename = os.path.splitext(os.path.basename(xml_file_path))[0]
        # Log with absolute index for clarity across runs
        current_file_log_prefix = f"File {current_xml_absolute_idx + 1}/{len(all_xml_files)} ({base_filename})"
        logging.info(f"--- {current_file_log_prefix}: Starting ---")

        extracted_text_file = os.path.join(EXTRACTED_TEXT_DIR, f"{base_filename}_extracted.txt")
        translated_text_file = os.path.join(TRANSLATED_TEXT_DIR, f"{base_filename}_translated_tr.txt")
        tts_audio_file = os.path.join(TTS_OUTPUT_DIR, f"{base_filename}_tts_tr.mp3")

        # Stage 1: Extraction
        extracted_text = None
        action_taken_extraction = False
        if os.path.exists(extracted_text_file):
            logging.info(f"{current_file_log_prefix}: Extracted text file found.")
            try:
                with open(extracted_text_file, "r", encoding="utf-8") as f:
                    extracted_text = f.read().strip()
                if not extracted_text:
                    logging.warning(f"{current_file_log_prefix}: Existing extracted file is empty. Re-extracting.")
                    os.remove(extracted_text_file)  # Remove empty file to force re-extraction
                    extracted_text = None
            except Exception as e:
                logging.error(f"{current_file_log_prefix}: Failed to read existing extracted file: {e}. Re-extracting.")
                extracted_text = None

        if extracted_text is None:
            logging.info(f"{current_file_log_prefix}: Extracting text...")
            extracted_text = extract_text_from_xml(xml_file_path)
            action_taken_extraction = True
            if extracted_text:
                try:
                    with open(extracted_text_file, "w", encoding="utf-8") as f:
                        f.write(extracted_text)
                    logging.info(f"{current_file_log_prefix}: Extracted text saved.")
                    newly_processed_stages_count += 1
                except Exception as e:
                    logging.error(f"{current_file_log_prefix}: Could not write extracted text: {e}")
                    extracted_text = None
            else:
                logging.warning(f"{current_file_log_prefix}: Extraction failed or produced no text. Skipping.")
                continue
        if not extracted_text: continue

        # Stage 2: Translation
        translated_text = None
        action_taken_translation = False
        if os.path.exists(translated_text_file):
            logging.info(f"{current_file_log_prefix}: Translated text file found.")
            try:
                with open(translated_text_file, "r", encoding="utf-8") as f:
                    translated_text = f.read().strip()
                if not translated_text:
                    logging.warning(f"{current_file_log_prefix}: Existing translated file is empty. Re-translating.")
                    os.remove(translated_text_file)
                    translated_text = None
            except Exception as e:
                logging.error(
                    f"{current_file_log_prefix}: Failed to read existing translated file: {e}. Re-translating.")
                translated_text = None

        if translated_text is None:
            logging.info(f"{current_file_log_prefix}: Translating text...")
            action_taken_translation = True
            translated_text_result = translate_text_gemini(extracted_text, ALL_GEMINI_MODELS_TO_TRY,
                                                           overall_translation_attempt_num)

            if translated_text_result == "RATE_LIMIT_RETRY":
                logging.critical(
                    f"{current_file_log_prefix}: Hit a rate limit on Gemini. Waiting 5 mins. Will retry this file in the next run if applicable.")
                time.sleep(300)
                overall_translation_attempt_num += 1
                continue
            elif translated_text_result:
                translated_text = translated_text_result
                try:
                    with open(translated_text_file, "w", encoding="utf-8") as f:
                        f.write(translated_text)
                    logging.info(f"{current_file_log_prefix}: Translated text saved.")
                    newly_processed_stages_count += 1
                    overall_translation_attempt_num = 0
                except Exception as e:
                    logging.error(f"{current_file_log_prefix}: Could not write translated text: {e}")
                    translated_text = None
            else:
                logging.error(f"{current_file_log_prefix}: Translation failed.")
        if not translated_text: continue

        # Stage 3: Text-to-Speech
        action_taken_tts = False
        if os.path.exists(tts_audio_file):
            logging.info(f"{current_file_log_prefix}: TTS audio file already exists.")
        else:
            logging.info(f"{current_file_log_prefix}: Synthesizing speech...")
            action_taken_tts = True
            if synthesize_speech_gcloud(translated_text, tts_audio_file):
                newly_processed_stages_count += 1
            else:  # TTS failed
                logging.error(f"{current_file_log_prefix}: TTS failed.")

        logging.info(f"--- {current_file_log_prefix}: Finished ---")

    logging.info(f"Run complete. Processed/attempted {newly_processed_stages_count} new file stages in this session.")


if __name__ == "__main__":
    logging.info(
        "Script started. Ensure GOOGLE_API_KEY (Gemini) and GOOGLE_APPLICATION_CREDENTIALS (Google TTS) are set.")

    # --- Batch processing ---
    # To process the first 5 XML files (from index 0 up to 5 files):
    # process_files_main(max_files_to_process_override=5, start_from_file_index=0)

    # To process the next 100 XML files (start from index 5, process up to 100 files from that point):
    # process_files_main(max_files_to_process_override=100, start_from_file_index=5)

    # To process all remaining XML files (e.g., if 105 total processed, start from index 105 and process all thereafter):
    # process_files_main(start_from_file_index=105) # max_files_to_process_override=None by default

    # Default: process all files from the beginning, or resume based on existing output files.
    process_files_main(max_files_to_process_override=None, start_from_file_index=0)

    logging.info("Script finished.")