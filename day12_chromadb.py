import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_documents")

model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample chunks (pretend these came from a real document)
chunks = [
    "Python is used for AI.",
    "SQL stores structured data.",
    "RAG combines retrieval and generation."
]

# Generate embeddings for all chunks
embeddings = model.encode(chunks).tolist()  # ChromaDB wants plain lists, not numpy arrays

# Add to ChromaDB — each item needs a unique id
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    embeddings=embeddings,
    documents=chunks
)

print("Added", collection.count(), "chunks to the database")

# Now query it
question = "What database stores structured data?"
question_embedding = model.encode([question]).tolist()

results = collection.query(
    query_embeddings=question_embedding,
    n_results=2   # get top 2 most relevant chunks
)

print("Top matches:", results["documents"])