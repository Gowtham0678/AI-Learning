from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
client=Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_about_document(document_text, question):
    sys=f"""You are a helpful assistant. Answer the user's question using ONLY the information in the document below. If the answer isn't in the document, say "I don't know based on this document."
    document:
    {document_text}"""
    response=client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"system","content":sys},{"role":"user","content":question}]
    )
    return response.choices[0].message.content
    

def load_document(filepath):
    with open(filepath,"r") as file:
        return file.read()
    
a=load_document("my_notes.txt")
print(ask_about_document(a,"Summarize this document in 2 lines"))
print(ask_about_document(a,"what is his goal?"))
print(ask_about_document(a,"what is his favorite place to visit?"))