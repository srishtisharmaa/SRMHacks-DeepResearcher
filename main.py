# deep_researcher_tfidf_improved.py
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

DATA_SOURCE = """
Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals. AI research has been defined as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. A key challenge is to create a system that can effectively gather, process, and retrieve relevant information.

Machine learning (ML) is a subset of AI that involves the use of algorithms to learn from data. Deep learning is a subfield of ML. The ability of a system to understand diverse codebases is part of the challenge in creating a general-purpose agent. The system must evaluate changes for quality and standards.

Natural language processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. Query refinement allows the user to ask follow-up questions to dig deeper into a topic. Multi-step reasoning is the process of breaking down a complex query into smaller, manageable tasks.

Robotics is an interdisciplinary field that integrates computer science, electrical engineering, and mechanical engineering. Robotics deals with the design, construction, operation, and use of robots. The goal of robotics is to create intelligent machines that can assist humans in a variety of ways.
"""

def get_corpus(data):
    """
    Splits the data into sentences robustly using regex.
    Keeps sentences reasonably sized and strips whitespace.
    """
    # Split on sentence end punctuation while keeping abbreviations somewhat safe.
    sentences = re.split(r'(?<=[.!?])\s+', data.strip())
    # Filter empty and trim
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def generate_local_embeddings(corpus):
    """
    Generates TF-IDF features for the corpus.
    Returns the fitted vectorizer and the embeddings matrix.
    """
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    corpus_embeddings = vectorizer.fit_transform(corpus)
    return vectorizer, corpus_embeddings

def retrieve_top_responses(query, vectorizer, corpus_embeddings, corpus, num_results=3):
    """
    Returns list of (sentence, score) for top num_results matches.
    """
    if not query or not query.strip():
        return []
    query_embedding = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_embedding, corpus_embeddings).flatten()
    top_indices = np.argsort(similarity_scores)[::-1][:num_results]
    results = []
    for idx in top_indices:
        score = float(similarity_scores[idx])
        text = corpus[idx]
        results.append((text, score))
    return results

def decompose_query(query):
    """
    Decompose on 'and', 'then', ';' or ',' heuristically (case-insensitive).
    Returns list of trimmed subqueries.
    """
    # normalize separators to a single token
    parts = re.split(r'\band\b|;|,|\bthen\b', query, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [query.strip()]

def handle_query(query, vectorizer, corpus_embeddings, corpus):
    """
    Multi-step reasoning: decompose then retrieve for each subquery.
    Returns structured text plus confidence hints.
    """
    subqueries = decompose_query(query)
    final_pieces = []
    for sub in subqueries:
        top = retrieve_top_responses(sub, vectorizer, corpus_embeddings, corpus, num_results=2)
        if not top:
            final_pieces.append(f"(No local match found for: '{sub}')")
            continue
        piece_lines = [f"Subquery: {sub}"]
        for i, (sent, score) in enumerate(top, start=1):
            piece_lines.append(f"  Match {i} (score={score:.3f}): {sent}")
        final_pieces.append("\n".join(piece_lines))
    return "\n\n".join(final_pieces)

def main():
    corpus = get_corpus(DATA_SOURCE)
    vectorizer, corpus_embeddings = generate_local_embeddings(corpus)

    print("Enhanced Deep Researcher Agent (TF-IDF prototype)")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("Enter your query: ").strip()
        if user_query.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        if not user_query:
            print("Please enter a non-empty query.")
            continue

        print("\nProcessing...\n")
        response = handle_query(user_query, vectorizer, corpus_embeddings, corpus)
        print("=== Research Summary ===")
        print(response)
        print("========================\n")

        # follow-up option
        follow = input("Ask a follow-up? (y/N): ").strip().lower()
        if follow != 'y':
            print("Returning to main prompt.\n")

if __name__ == "__main__":
    main()
