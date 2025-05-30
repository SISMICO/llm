#!/usr/bin/env python3

import os
import requests
import json
from dotenv import load_dotenv

# Langchain and RAG specific imports
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings, ChatOllama # For RAG
from langchain.agents import tool, AgentExecutor, create_react_agent
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain

# FAISS and Numpy for vector search
import faiss
import numpy as np
# --- Configuration ---
# Ensure your FastAPI application (api.py) is running, typically on this URL.
API_BASE_URL = "http://127.0.0.1:8000"

# Specify the Ollama model you want to use.
# Make sure Ollama server is running (`ollama serve`)
# and the model is pulled (e.g., `ollama pull llama3` or `ollama pull mistral`)
load_dotenv() # Load environment variables from .env file

# Agent's LLM
AGENT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2") # Or "mistral", "llama2", etc.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# RAG specific configurations
RAG_OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "")
RAG_OLLAMA_LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Attempt to import FAISS_INDEX_PATH and DOC_METADATA_PATH from faiss_utils
# If not found, define defaults. Ensure faiss_utils.py is in src/
try:
    from src.faiss_utils import FAISS_INDEX_PATH, DOC_METADATA_PATH, load_faiss_index
except ImportError:
    print("Warning: src.faiss_utils not found or paths not defined. Using default FAISS paths.")
    FAISS_INDEX_PATH = "faiss_index.idx"
    DOC_METADATA_PATH = "doc_metadata.json"
    # Define a placeholder load_faiss_index if not importable, or handle error
    def load_faiss_index(FAISS_INDEX_PATH, DOC_METADATA_PATH):
        print(f"Error: FAISS utility 'load_faiss_index' not available from src.faiss_utils.")
        print(f"Please ensure src/faiss_utils.py exists and defines this function, FAISS_INDEX_PATH, and DOC_METADATA_PATH.")
        return None, None

# --- Tool Definitions for Langchain Agent ---
# These tools correspond to your FastAPI endpoints.
# The docstrings are VERY important as they tell the LLM how and when to use the tool.

@tool
def sum_numbers_tool(numbers_str: str) -> str:
    """
    Useful for when you need to sum two numbers.
    The input to this tool should be a comma-separated string of two numbers.
    For example, `5,3` would be the input if you wanted to sum 5 and 3.
    """
    print(f"\n>>> [Tool Call] sum_numbers_tool(input='{numbers_str}')")
    try:
        num1_str, num2_str = numbers_str.split(',')
        number1 = float(num1_str.strip())
        number2 = float(num2_str.strip())
    except ValueError:
        error_msg = "Invalid input format for sum_numbers_tool. Please provide two numbers separated by a comma (e.g., '5,3')."
        print(f"<<< [Tool Response] Error: {error_msg}")
        return error_msg

    try:
        response = requests.post(f"{API_BASE_URL}/sum", json={"number1": number1, "number2": number2})
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        result = response.json().get("result")
        output = f"The sum of {number1} and {number2} is {result}."
        print(f"<<< [Tool Response] {output}")
        return output
    except requests.exceptions.RequestException as e:
        error_msg = f"Error calling sum API: {e}"
        print(f"<<< [Tool Response] Error: {error_msg}")
        return error_msg
    except json.JSONDecodeError:
        error_msg = "Error: Could not decode JSON response from sum API."
        print(f"<<< [Tool Response] Error: {error_msg}")
        return error_msg


@tool
def multiply_numbers_tool(numbers_str: str) -> str:
    """
    Useful for when you need to multiply two numbers.
    The input to this tool should be a comma-separated string of two numbers.
    For example, `5,3` would be the input if you wanted to multiply 5 and 3.
    """
    print(f"\n>>> [Tool Call] multiply_numbers_tool(input='{numbers_str}')")
    try:
        num1_str, num2_str = numbers_str.split(',')
        number1 = float(num1_str.strip())
        number2 = float(num2_str.strip())
    except ValueError:
        error_msg = "Invalid input format for multiply_numbers_tool. Please provide two numbers separated by a comma (e.g., '5,3')."
        print(f"<<< [Tool Response] Error: {error_msg}")
        return error_msg

    try:
        response = requests.post(f"{API_BASE_URL}/multiply", json={"number1": number1, "number2": number2})
        response.raise_for_status()
        result = response.json().get("result")
        output = f"The product of {number1} and {number2} is {result}."
        print(f"<<< [Tool Response] {output}")
        return output
    except requests.exceptions.RequestException as e:
        error_msg = f"Error calling multiply API: {e}"
        print(f"<<< [Tool Response] Error: {error_msg}")
        return error_msg
    except json.JSONDecodeError:
        error_msg = "Error: Could not decode JSON response from multiply API."
        print(f"<<< [Tool Response] Error: {error_msg}")
        return error_msg


@tool
def get_person_info_tool(person_id_str: str) -> str:
    """
    Useful for when you need to retrieve information about a person using their ID.
    The input to this tool should be a single integer representing the person's ID.
    For example, `1` would be the input if you wanted information for person ID 1.
    """
    print(f"\n>>> [Tool Call] get_person_info_tool(input='{person_id_str}')")
    try:
        print(f"Number Received: {person_id_str}")
        person_id = int(person_id_str.strip('`'))
    except ValueError:
        error_msg = "Invalid input format for get_person_info_tool. Please provide a single integer ID (e.g., '1')."
        print(f"<<< [Tool Response] Error: {error_msg}")
        return error_msg

    try:
        response = requests.get(f"{API_BASE_URL}/person/{person_id}")
        if response.status_code == 404:
            error_msg = f"Person with ID {person_id} not found."
            print(f"<<< [Tool Response] {error_msg}")
            return error_msg
        response.raise_for_status()  # Raise for other HTTP errors
        person_info = response.json()
        output = f"Information for person ID {person_id}: {json.dumps(person_info)}"
        print(f"<<< [Tool Response] {output}")
        return output
    except requests.exceptions.RequestException as e:
        error_msg = f"Error calling person API: {e}"
        print(f"<<< [Tool Response] Error: {error_msg}")
        return error_msg
    except json.JSONDecodeError:
        error_msg = "Error: Could not decode JSON response from person API."
        print(f"<<< [Tool Response] Error: {error_msg}")
        return error_msg

# --- RAG Tool: Query Knowledge Base ---

# Global variables for RAG resources
rag_embedding_model_instance = None
rag_llm_instance = None
rag_faiss_index_instance = None
rag_doc_metadata_instance = None
rag_resources_initialized = False

def initialize_rag_resources():
    """Initializes resources for the RAG knowledge base tool."""
    global rag_embedding_model_instance, rag_llm_instance, rag_faiss_index_instance, rag_doc_metadata_instance, rag_resources_initialized

    if rag_resources_initialized:
        return True

    print("\n>>> Initializing RAG (Knowledge Base) Resources...")
    try:
        # Initialize Embedding Model for RAG
        print(f"Initializing Ollama embedding model for RAG: {RAG_OLLAMA_EMBEDDING_MODEL} from {OLLAMA_BASE_URL}...")
        rag_embedding_model_instance = OllamaEmbeddings(model=RAG_OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        rag_embedding_model_instance.embed_query("test embedding") # Test
        print("Ollama embedding model for RAG initialized.")

        # Load FAISS Index and Metadata
        print(f"Loading FAISS index from: {FAISS_INDEX_PATH}")
        print(f"Loading document metadata from: {DOC_METADATA_PATH}")
        rag_faiss_index_instance, rag_doc_metadata_instance = load_faiss_index(FAISS_INDEX_PATH, DOC_METADATA_PATH)
        if rag_faiss_index_instance is None or not rag_doc_metadata_instance:
            print("WARNING: FAISS index or metadata not loaded. Knowledge base tool will not be functional.")
            # Do not set rag_resources_initialized to True if essential parts fail
            return False
        print(f"FAISS index (ntotal={rag_faiss_index_instance.ntotal}) and metadata (count={len(rag_doc_metadata_instance)}) loaded.")

        # Initialize LLM for RAG Answer Generation
        print(f"Initializing Ollama LLM for RAG generation: {RAG_OLLAMA_LLM_MODEL} from {OLLAMA_BASE_URL}...")
        rag_llm_instance = ChatOllama(model=RAG_OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL)
        rag_llm_instance.invoke("Respond with 'ok' if you are working.") # Test
        print("Ollama LLM for RAG generation initialized.")

        rag_resources_initialized = True
        print(">>> RAG (Knowledge Base) Resources Initialized Successfully.")
        return True
    except Exception as e:
        print(f"Error initializing RAG resources: {e}")
        print("Knowledge base tool may not be functional.")
        return False

@tool
def query_knowledge_base_tool(user_query: str) -> str:
    """
    Useful for when you need to answer questions based on a knowledge base of documents.
    Use this tool if the user's question seems to require information that might be found in specific documents,
    such as 'What are the features of X?', 'Summarize Y based on our documents', or 'Tell me about Z'.
    Do NOT use this tool for calculations or general knowledge questions if other tools are more appropriate.
    The input should be the user's full question.
    """
    print(f"\n>>> [Tool Call] query_knowledge_base_tool(input='{user_query}')")

    if not rag_resources_initialized:
        error_msg = "Knowledge base resources are not initialized. Cannot answer the query."
        print(f"<<< [Tool Response] Error: {error_msg}")
        return error_msg
    
    if not all([rag_embedding_model_instance, rag_faiss_index_instance, rag_doc_metadata_instance, rag_llm_instance]):
        error_msg = "One or more RAG components are missing. Cannot process the query."
        print(f"<<< [Tool Response] Error: {error_msg}")
        return error_msg

    # 1. Find similar documents
    try:
        query_embedding = rag_embedding_model_instance.embed_query(user_query)
        query_embedding_np = np.array([query_embedding], dtype='float32')
        faiss.normalize_L2(query_embedding_np) # Normalize if index embeddings were normalized

        distances, indices = rag_faiss_index_instance.search(query_embedding_np, k=3) # Retrieve top 3
        
        retrieved_docs_data = []
        if indices.size > 0:
            for i, idx in enumerate(indices[0]):
                if idx != -1 and 0 <= idx < len(rag_doc_metadata_instance):
                    metadata = rag_doc_metadata_instance[idx]
                    retrieved_docs_data.append(Document(page_content=metadata["content"], metadata={"source": metadata["source"], "chunk_index": metadata["chunk_index"]}))
        print(f"Found {len(retrieved_docs_data)} relevant document chunks.")
    except Exception as e:
        error_msg = f"Error during similarity search: {e}"
        print(f"<<< [Tool Response] Error: {error_msg}")
        return error_msg

    if not retrieved_docs_data:
        output = "I couldn't find any relevant documents in the knowledge base to answer your question."
        print(f"<<< [Tool Response] {output}")
        return output

    # 2. Generate contextual answer
    try:
        prompt_template_str = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer based on the context, state that you don't know.
Do not make up information outside of the provided context. Keep the answer concise.
Context: {context}
Question: {input}
Answer:"""
        prompt = ChatPromptTemplate.from_template(prompt_template_str)
        qa_chain = create_stuff_documents_chain(rag_llm_instance, prompt)
        response = qa_chain.invoke({"input": user_query, "context": retrieved_docs_data})
        print(f"<<< [Tool Response] {response}")
        return response
    except Exception as e:
        error_msg = f"Error during RAG answer generation: {e}"
        print(f"<<< [Tool Response] Error: {error_msg}")
        return error_msg

# --- Langchain Agent Setup ---
def run_agent():
    """
    Sets up and runs the Langchain agent.
    """
    tools = [sum_numbers_tool, multiply_numbers_tool, get_person_info_tool, query_knowledge_base_tool]

    try:
        # Initialize the Ollama LLM
        llm = OllamaLLM(model=AGENT_OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        # Test connection (optional, but good for early failure)
        llm.invoke("Hello!")
        print(f"Successfully connected to Agent Ollama model: {AGENT_OLLAMA_MODEL}")
    except Exception as e:
        print(f"Error initializing Agent Ollama LLM ({AGENT_OLLAMA_MODEL}): {e}")
        print("Please ensure the Ollama server is running and the model is available.")
        print(f"You can run 'ollama serve' and 'ollama pull {AGENT_OLLAMA_MODEL}'.")
        return

    # Initialize RAG resources
    # If RAG resources fail to initialize, the agent can still run,
    # but the query_knowledge_base_tool will return an error if called.
    initialize_rag_resources()

    # Get the ReAct prompt from Langchain Hub
    # This prompt helps the LLM decide when to use tools and how to reason.
    try:
        prompt = hub.pull("hwchase17/react")
    except Exception as e:
        print(f"Error pulling ReAct prompt from Langchain Hub: {e}")
        print("Please ensure you have internet connectivity and 'langchainhub' is installed.")
        return

    # Create the ReAct agent
    agent = create_react_agent(llm, tools, prompt)

    # Create an agent executor
    # verbose=True will print the agent's thought process, which is very helpful for debugging.
    # handle_parsing_errors=True helps if the LLM doesn't format its output perfectly for tool use.
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors="Check your output and make sure it conforms to the Action/Action Input format.",
        max_iterations=5 # Prevents overly long loops
    )

    print("\n--- LLM Powered Assistant ---")
    print(f"Using Agent Ollama model: {AGENT_OLLAMA_MODEL}")
    print("I can perform calculations, get person information, or answer questions based on a knowledge base.")
    print("Type 'exit' or 'quit' to end.")

    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting assistant...")
                break

            if not user_input.strip():
                continue

            # Invoke the agent with the user's input
            response = agent_executor.invoke({"input": user_input})

            print(f"\nAssistant: {response['output']}")

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            # Depending on the error, you might want to break or allow continuation

if __name__ == "__main__":
    # Before running, ensure:
    # 1. Your FastAPI application (`api.py`) is running.
    #    (e.g., `uvicorn api:app --app-dir src --reload` from your project root)
    # 2. Ollama server is running (`ollama serve`).
    # 3. The specified Ollama models (AGENT_OLLAMA_MODEL, RAG_OLLAMA_EMBEDDING_MODEL, RAG_OLLAMA_LLM_MODEL) are pulled.
    #    (e.g., `ollama pull llama3.2`, `ollama pull mxbai-embed-large`)
    # 4. FAISS index (faiss_index.idx) and metadata (doc_metadata.json) are present (usually in the root or `src` directory, configure paths as needed).
    # 5. Required Python packages are installed:
    #    `pip install requests langchain langchain-community langchainhub ollama python-dotenv faiss-cpu numpy`

    run_agent()
