# file: 01_sanity_check.py
# This file checks if the connection to the OpenAI model works properly.

from langchain_openai import ChatOpenAI


def main():

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Send a message to the model and store the reply in 'resp'.
    resp = llm.invoke("Say 'hello' and tell me your model name.")
    print("✅ LLM OK:", resp.content)


if __name__ == "__main__":
    main()
