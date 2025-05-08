#!/usr/bin/env python3

import os
import requests
import json
from langchain_ollama import OllamaLLM
from langchain.agents import tool, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain import hub

# --- Configuration ---
# Ensure your FastAPI application (api.py) is running, typically on this URL.
API_BASE_URL = "http://127.0.0.1:8000"

# Specify the Ollama model you want to use.
# Make sure Ollama server is running (`ollama serve`)
# and the model is pulled (e.g., `ollama pull llama3` or `ollama pull mistral`)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2") # Or "mistral", "llama2", etc.


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
        person_id = int(person_id_str.strip())
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


# --- Langchain Agent Setup ---
def run_agent():
    """
    Sets up and runs the Langchain agent.
    """
    tools = [sum_numbers_tool, multiply_numbers_tool, get_person_info_tool]

    try:
        # Initialize the Ollama LLM
        llm = OllamaLLM(model=OLLAMA_MODEL)
        # Test connection (optional, but good for early failure)
        llm.invoke("Hello!")
        print(f"Successfully connected to Ollama model: {OLLAMA_MODEL}")
    except Exception as e:
        print(f"Error initializing Ollama LLM ({OLLAMA_MODEL}): {e}")
        print("Please ensure the Ollama server is running and the model is available.")
        print("You can run 'ollama serve' and 'ollama pull {OLLAMA_MODEL}'.")
        return

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
        verbose=True,
        handle_parsing_errors="Check your output and make sure it conforms to the Action/Action Input format.",
        max_iterations=5 # Prevents overly long loops
    )

    print("\n--- LLM Powered Assistant ---")
    print(f"Using Ollama model: {OLLAMA_MODEL}")
    print("I can sum numbers, multiply numbers, or get person information by ID.")
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
    # 3. The specified OLLAMA_MODEL is pulled (`ollama pull llama3`).
    # 4. Required Python packages are installed:
    #    `pip install requests langchain langchain-community langchainhub ollama`

    run_agent()
