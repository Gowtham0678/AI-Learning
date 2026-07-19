import os
import chromadb
import hashlib
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from pypdf import PdfReader


load_dotenv()

# --- Setup ---
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="my_documents")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_file_hash(filepath):
    """Returns MD5 hash of a file."""

    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
    
def load_and_store_document(filepath):
    """Reads a document, chunks it, embeds it, and stores it."""

    if filepath.endswith(".pdf"):
        text = load_pdf_text(filepath)
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

    if not text.strip():
        print("No text found.")
        return

    file_hash = get_file_hash(filepath)

    existing = collection.get(
        where={"file_hash": file_hash}
    )

    if len(existing["ids"]) > 0:
        print("Document already exists in ChromaDB.")
        return

    chunks = chunk_text(text)

    embeddings = embed_model.encode(chunks).tolist()

    base_name = os.path.basename(filepath)

    ids = []

    metadata = []

    for i in range(len(chunks)):
        ids.append(f"{base_name}_{i}")

        metadata.append({
            "source": base_name,
            "chunk": i,
            "file_hash": file_hash
        })

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadata
    )

    print(f"{len(chunks)} chunks stored.")

def rewrite_query(question):

    prompt = f"""
Rewrite this question to improve semantic search.

Question:

{question}

Return only the rewritten question.
"""

    response = groq_client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "system",
                "content": "You rewrite search queries."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content.strip()


def load_pdf_text(filepath):
    """Extracts all text from a PDF file and returns it as one string."""
    reader = PdfReader(filepath)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text


def chunk_text(text, chunk_size=500, overlap=50):
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def load_and_store_document(filepath):
    """Reads a .txt or .pdf file, chunks it, embeds it, and stores it in ChromaDB."""
    if filepath.endswith(".pdf"):
        text = load_pdf_text(filepath)
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

    if not text.strip():
        print("Warning: no text could be extracted from this file.")
        return

    chunks = chunk_text(text)
    embeddings = embed_model.encode(chunks).tolist()

    # Use filename as prefix so multiple documents don't collide on ids
    base_name = os.path.basename(filepath)
    ids = [f"{base_name}_chunk_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks
    )
    print(f"Stored {len(chunks)} chunks from {filepath}")


def ask_question(question, n_results=3):

    better_question = rewrite_query(question)

    print("\nOriginal Question:")
    print(question)

    print("\nRewritten Question:")
    print(better_question)

    question_embedding = embed_model.encode(
        [better_question]
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=n_results
    )

    retrieved_chunks = results["documents"][0]
    metadata = results["metadatas"][0]

    print("\nRetrieved Chunks")
    print("=" * 60)

    for i, (chunk, meta) in enumerate(zip(retrieved_chunks, metadata), start=1):

        print(f"\nChunk {i}")

        print(meta)

        print(chunk[:250])

        print("-" * 60)

    context = "\n".join(retrieved_chunks)

    system_prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.

If the answer is not present,
reply:

"I don't have that information based on this document."

Context:

{context}
"""

    response = groq_client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.choices[0].message.content   


# --- Run it ---
if __name__ == "__main__":

    load_and_store_document("Gowtham_AIML_Roadmap.pdf")

    print("\nRAG Chatbot Ready")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("Ask: ")

        if question.lower() == "exit":
            break

        answer = ask_question(question)

        print("\nAnswer:\n")

        print(answer)

        print("\n" + "=" * 60)    