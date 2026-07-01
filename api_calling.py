from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

conversation_history = []

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
    print("AI Career Assistant (type 'exit' to quit)")
    print("-" * 40)
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Goodbye! Good luck with your AI journey 🚀")
            break
        try:
            response = chat(user_input)
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"Error: {e}")

run_chatbot()