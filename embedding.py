from sentence_transformers import SentenceTransformer
from sentence_transformers import util


# Load a small, fast, free embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "The cat sat on the mat",
    "A feline rested on the rug",
    "Stock prices rose today"
]

embeddings = model.encode(sentences)
question = "What pets are mentioned?"
question_embedding = model.encode([question])[0]

# Compare question against all your document chunks
similarities = util.cos_sim(question_embedding, embeddings)[0]

# Find the most similar chunk
best_match_index = similarities.argmax()
print("Most relevant chunk:", sentences[best_match_index])
print("Similarity score:", similarities[best_match_index])