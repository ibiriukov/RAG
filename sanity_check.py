# file: 01_sanity_check.py
# This file checks if the connection to the OpenAI model works properly.

# Import the ChatOpenAI class from the LangChain library.
# It lets Python communicate with OpenAI’s GPT models (like GPT-4o-mini).
from langchain_openai import ChatOpenAI


# Define the main() function — this is where our main code lives.
def main():
    # Create a ChatOpenAI object called 'llm' to talk to the GPT-4o-mini model.
    # temperature=0 means no randomness, so results are consistent every time.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Send a message to the model and store the reply in 'resp'.
    resp = llm.invoke("Say 'hello' and tell me your model name.")

    # Print the model’s response to confirm that everything is working.
    print("✅ LLM OK:", resp.content)


# This special check runs only if this file is executed directly (not imported).
if __name__ == "__main__":
    # Call the main() function to start the test.
    main()
