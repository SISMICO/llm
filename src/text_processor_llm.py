#!/usr/bin/env python3

import os
import logging
import re
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Configuration for Ollama
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2") # Or your preferred model
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm_instance = None

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

def initialize_llm():
    """Initializes the Ollama LLM instance."""
    global llm_instance
    if llm_instance is None:
        try:
            logging.info(f"Initializing Ollama model: {OLLAMA_MODEL_NAME} from {OLLAMA_BASE_URL}...")
            llm_instance = ChatOllama(
                model=OLLAMA_MODEL_NAME, 
                base_url=OLLAMA_BASE_URL,
                temperature=0.1,  # Lower temperature for more focused responses
            )
            # Test LLM to ensure it's working
            llm_instance.invoke("Respond with 'ok' if you are working.")
            logging.info(f"Ollama model '{OLLAMA_MODEL_NAME}' initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Ollama model '{OLLAMA_MODEL_NAME}': {e}")
            logging.error("Please ensure Ollama is running and the model is available.")
            logging.error(f"You can try: \n1. Run 'ollama serve' in your terminal. \n2. Pull the model: 'ollama pull {OLLAMA_MODEL_NAME}'")
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

def main():
    try:
        initialize_llm()
    except Exception:
        print("Exiting due to LLM initialization error. Check logs.")
        return

    print(f"\n--- Text Processor (Model: {OLLAMA_MODEL_NAME}) ---")
    print("Enter English text to correct and translate. Type 'exit' or 'quit' to end.")

    while True:
        try:
            user_input = input("\nEnglish Text: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting processor...")
                break

            if not user_input.strip():
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

        except Exception as e:
            logging.error(f"An unexpected error occurred in the main loop: {e}")

if __name__ == "__main__":
    main()
