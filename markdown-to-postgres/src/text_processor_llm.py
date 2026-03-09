#!/usr/bin/env python3

import os
import logging
import re
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Configuration for Ollama
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2") # Or your preferred model
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Configuration for Gemini
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

llm_instance = None
current_model_type = None

def clean_response(text: str) -> str:
    """Remove thinking tags and other unwanted markup from LLM response."""
    # Remove thinking tags and their content
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove other common thinking patterns
    text = re.sub(r'\*\*Thinking:?\*\*.*?(?=\n\n|\n[A-Z]|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Thinking:.*?(?=\n\n|\n[A-Z]|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove introductory phrases that might have slipped through
    text = re.sub(r'^(Here is the corrected text:?|Here is the translation:?|Corrected text:?|Translation:?)\s*', '', text, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n', '\n', text)
    text = text.strip()
    
    return text

def initialize_llm(model_type="ollama"):
    """Initializes the LLM instance (Ollama or Gemini)."""
    global llm_instance, current_model_type
    
    # If already initialized with the same model type, return existing instance
    if llm_instance is not None and current_model_type == model_type:
        return llm_instance
    
    # Reset instance if switching models
    llm_instance = None
    current_model_type = model_type
    
    try:
        if model_type.lower() == "ollama":
            logging.info(f"Initializing Ollama model: {OLLAMA_MODEL_NAME} from {OLLAMA_BASE_URL}...")
            llm_instance = ChatOllama(
                model=OLLAMA_MODEL_NAME, 
                base_url=OLLAMA_BASE_URL,
                temperature=0.1,  # Lower temperature for more focused responses
            )
            # Test LLM to ensure it's working
            llm_instance.invoke("Respond with 'ok' if you are working.")
            logging.info(f"Ollama model '{OLLAMA_MODEL_NAME}' initialized successfully.")
            
        elif model_type.lower() == "gemini":
            if not GEMINI_API_KEY:
                raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
            
            logging.info(f"Initializing Gemini model: {GEMINI_MODEL_NAME}...")
            llm_instance = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL_NAME,
                google_api_key=GEMINI_API_KEY,
                temperature=0.1,  # Lower temperature for more focused responses
            )
            # Test LLM to ensure it's working
            llm_instance.invoke("Respond with 'ok' if you are working.")
            logging.info(f"Gemini model '{GEMINI_MODEL_NAME}' initialized successfully.")
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Use 'ollama' or 'gemini'.")
            
    except Exception as e:
        if model_type.lower() == "ollama":
            logging.error(f"Failed to initialize Ollama model '{OLLAMA_MODEL_NAME}': {e}")
            logging.error("Please ensure Ollama is running and the model is available.")
            logging.error(f"You can try: \n1. Run 'ollama serve' in your terminal. \n2. Pull the model: 'ollama pull {OLLAMA_MODEL_NAME}'")
        elif model_type.lower() == "gemini":
            logging.error(f"Failed to initialize Gemini model '{GEMINI_MODEL_NAME}': {e}")
            logging.error("Please ensure your GOOGLE_API_KEY is set correctly in your .env file.")
        raise
        
    return llm_instance

def correct_grammar_and_syntax(text: str) -> str:
    """Corrects grammar and syntax of the input English text using the LLM."""
    if not llm_instance:
        logging.error("LLM not initialized. Cannot correct text.")
        return "Error: LLM not initialized."

    prompt_template_str = """You are an expert English proofreader. Your task is to ONLY provide the corrected text.

CRITICAL INSTRUCTIONS - FOLLOW EXACTLY:
- Do NOT use any XML tags like <thinking>, <think>, or any other markup
- Do NOT explain what you changed
- Do NOT provide reasoning or thinking steps
- Do NOT add any commentary, analysis, or notes
- Do NOT include phrases like "Here is the corrected text:" or similar
- Do NOT use any formatting tags or markup
- ONLY output the corrected text itself, nothing else

Correct any grammatical errors and improve the syntax while keeping the original meaning:

{english_text}"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    chain = prompt | llm_instance | StrOutputParser()

    try:
        logging.info("Requesting grammar and syntax correction from LLM...")
        corrected_text = chain.invoke({"english_text": text})
        return clean_response(corrected_text)
    except Exception as e:
        logging.error(f"Error during grammar correction: {e}")
        return f"Error during grammar correction: {e}"

def translate_to_portuguese_br(text: str) -> str:
    """Translates the input English text to Brazilian Portuguese using the LLM."""
    if not llm_instance:
        logging.error("LLM not initialized. Cannot translate text.")
        return "Error: LLM not initialized."

    prompt_template_str = """You are an expert translator. Your task is to ONLY provide the translation.

CRITICAL INSTRUCTIONS - FOLLOW EXACTLY:
- Do NOT use any XML tags like <thinking>, <think>, or any other markup
- Do NOT explain your translation choices
- Do NOT provide reasoning or thinking steps
- Do NOT add any commentary, analysis, or notes
- Do NOT include phrases like "Here is the translation:" or similar
- Do NOT use any formatting tags or markup
- ONLY output the Brazilian Portuguese translation itself, nothing else

Translate this English text to Brazilian Portuguese:

{english_text_to_translate}"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    chain = prompt | llm_instance | StrOutputParser()

    try:
        logging.info("Requesting translation to Brazilian Portuguese from LLM...")
        translated_text = chain.invoke({"english_text_to_translate": text})
        return clean_response(translated_text)
    except Exception as e:
        logging.error(f"Error during translation: {e}")
        return f"Error during translation: {e}"

def select_model() -> str:
    """Let user select which model to use."""
    print("\nSelect the AI model to use:")
    print("1. Ollama (Local)")
    print("2. Gemini (Google)")
    
    while True:
        try:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            if choice == "1":
                return "ollama"
            elif choice == "2":
                return "gemini"
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit(0)

def get_multiline_input() -> str:
    """Get multi-line input from user until they type 'END' on a separate line."""
    print("Enter your English text (multiple lines allowed).")
    print("Type 'END' on a new line when finished, or press Ctrl+D:")
    print("-" * 50)
    
    lines = []
    try:
        while True:
            try:
                line = input()
                if line.strip().upper() == 'END' or line.strip().upper() == '.':
                    break
                lines.append(line)
            except EOFError:  # Ctrl+D pressed
                break
    except KeyboardInterrupt:  # Ctrl+C pressed
        return ""
    
    return '\n'.join(lines).strip()

def main():
    print("--- Text Processor with AI Models ---")
    print("Process English text with grammar correction and translation.")
    
    # Let user select the model
    selected_model = select_model()
    
    try:
        initialize_llm(selected_model)
    except Exception:
        print(f"Exiting due to {selected_model.title()} initialization error. Check logs.")
        return

    model_display_name = f"{GEMINI_MODEL_NAME}" if selected_model == "gemini" else f"{OLLAMA_MODEL_NAME}"
    print(f"\n--- Text Processor (Model: {model_display_name}) ---")
    print("Type 'exit' or 'quit' to end the program.")
    print("Type 'switch' to change the AI model.")
    print("\nInstructions for multi-line input:")
    print("- You can enter multiple paragraphs")
    print("- Type 'END' on a new line when finished")
    print("- Or press Ctrl+D (Ctrl+Z on Windows) to finish input")

    while True:
        try:
            print("\n" + "="*60)
            user_input = get_multiline_input()
            
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting processor...")
                break
            elif user_input.lower() == 'switch':
                print("\nSwitching model...")
                selected_model = select_model()
                try:
                    initialize_llm(selected_model)
                    model_display_name = f"{GEMINI_MODEL_NAME}" if selected_model == "gemini" else f"{OLLAMA_MODEL_NAME}"
                    print(f"Switched to {model_display_name}")
                    continue
                except Exception:
                    print(f"Failed to switch to {selected_model.title()}. Check logs.")
                    continue

            if not user_input.strip():
                print("No text entered. Please try again.")
                continue

            print("\nProcessing...")
            corrected_text = correct_grammar_and_syntax(user_input)
            
            print(f"\n--- Original English Text ---")
            print(user_input)
            
            print(f"\n--- Corrected English Text ---")
            print(corrected_text)

            # translated_text_pt_br = translate_to_portuguese_br(corrected_text) # Translate the corrected text
            # print(f"\n--- Brazilian Portuguese Translation ---")
            # print(translated_text_pt_br)
            # print("\n--------------------------------------")

        except KeyboardInterrupt:
            print("\nExiting processor...")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred in the main loop: {e}")
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
