from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

def main():
    # To ensure Ollama is running, you can execute `ollama serve` or `ollama run llama3.2` in your terminal.
    model_name = "llama3.2" # Replace with your desired Ollama model

    try:
        llm = ChatOllama(model=model_name)
        llm.invoke("Test query: Is Ollama working?", stop=["\n"])
        print(f"Successfully connected to Ollama and model '{model_name}'.")
    except Exception as e:
        print(f"Error initializing Ollama or connecting to the model '{model_name}': {e}")
        print("Please ensure Ollama is running and the specified model is available.")
        print(f"You can try: \n1. Run 'ollama serve' in your terminal. \n2. Pull the model: 'ollama pull {model_name}'")
        return

    # --- 2. Define the Prompt Template ---
    # This template structures the input for the LLM.
    prompt_template_str = """
Human: {question}

AI Assistant:"""

    prompt = PromptTemplate.from_template(prompt_template_str)

    # --- 3. Create the Chain using Langchain Expression Language (LCEL) ---
    # This chain will take the user's question, format it with the prompt,
    # send it to the LLM, and parse the output as a string.
    chain = prompt | llm | StrOutputParser()

    # --- 4. Get user input ---
    print("\nAsk a question to the local LLM (type 'exit' to quit).")
    while True:
        user_question = input("\nYour question: ")

        if user_question.strip().lower() == 'exit':
            print("Exiting program.")
            break
        
        if not user_question.strip():
            print("No question provided. Please enter a question or type 'exit'.")
            continue

        # --- 5. Compute the answer ---
        print("\nThinking...")
        try:
            answer = chain.invoke({"question": user_question})
            # --- 6. Say the answer to the prompt (print to console) ---
            print("\nLLM's Answer:")
            print(answer)
        except Exception as e:
            print(f"Error during LLM query: {e}")

if __name__ == "__main__":
    main()
