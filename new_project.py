from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask_about_document(document_text, question):
    system_prompt = f"""You are a helpful assistant. Answer the user's question 
using ONLY the information in the document below. If the answer isn't in the 
document, say "I don't know based on this document."

Document:
{document_text}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content


def load_document(filepath):
    with open(filepath, "r") as file:
        return file.read()


# ---------- This is what actually runs ----------

document_text = load_document("my_notes.txt")

answer = ask_about_document(document_text, "Summarize this document in 2 lines")
print(answer)