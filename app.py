from dotenv import load_dotenv
import os
from groq import Groq
import json

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask_ai(question, system_prompt="You are a helpful assistant."):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content


def load_conversation():
    try:
        with open("chat_history.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []


def save_conversation():
    with open("chat_history.json", "w") as file:
        json.dump(conversation_history, file, indent=2)


def chat(question):
    conversation_history.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=conversation_history
    )

    answer = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": answer})

    return answer


def run_chatbot():
    print("Chatbot ready! Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            save_conversation()
            print("Saved! Goodbye 👋")
            break

        answer = chat(user_input)
        print(f"Assistant: {answer}")


# ---------- Everything below this line actually RUNS ----------

conversation_history = load_conversation()   # set up memory before chatting

run_chatbot()   # start the chatbot loop