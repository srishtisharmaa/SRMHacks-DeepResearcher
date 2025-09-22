from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# A local data source for the researcher agent.
DATA_SOURCE = """
Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals. AI research has been defined as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. A key challenge is to create a system that can effectively gather, process, and retrieve relevant information.

Machine learning (ML) is a subset of AI that involves the use of algorithms to learn from data. Deep learning is a subfield of ML. The ability of a system to understand diverse codebases is part of the challenge in creating a general-purpose agent. The system must evaluate changes for quality and standards.

Natural language processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. Query refinement allows the user to ask follow-up questions to dig deeper into a topic. Multi-step reasoning is the process of breaking down a complex query into smaller, manageable tasks.

Robotics is an interdisciplinary field that integrates computer science, electrical engineering, and mechanical engineering. Robotics deals with the design, construction, operation, and use of robots. The goal of robotics is to create intelligent machines that can assist humans in a variety of ways.
"""

def get_corpus(data):
    """Splits the data source into sentences to create a corpus for embedding."""
    return data.split('. ')

def generate_local_embeddings(corpus):
    """
    Generates local embeddings using TF-IDF and fits the vectorizer to the data.
    """
    vectorizer = TfidfVectorizer()
    corpus_embeddings = vectorizer.fit_transform(corpus)
    return vectorizer, corpus_embeddings

def retrieve_top_responses(query, vectorizer, corpus_embeddings, corpus, num_results=3):
    """
    Retrieves the top N most relevant responses from the corpus.
    """
    query_embedding = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_embedding, corpus_embeddings).flatten()
    
    # Get the indices of the top N most relevant sentences
    top_indices = np.argsort(similarity_scores)[::-1][:num_results]
    
    # Return the corresponding sentences
    return [corpus[i] for i in top_indices]

def handle_query(query, vectorizer, corpus_embeddings, corpus):
    """
    Processes a user query, now with improved multi-step reasoning.
    """
    # Simple multi-step reasoning: Split the query by "and"
    if " and " in query.lower():
        sub_queries = query.split(" and ")
        results = []
        for sub_q in sub_queries:
            top_results = retrieve_top_responses(sub_q, vectorizer, corpus_embeddings, corpus, num_results=1)
            results.append(top_results[0])
        return " and ".join(results)
    else:
        return "\n".join(retrieve_top_responses(query, vectorizer, corpus_embeddings, corpus))

def main():
    """Main function to run the Deep Researcher Agent."""
    corpus = get_corpus(DATA_SOURCE)
    vectorizer, corpus_embeddings = generate_local_embeddings(corpus)
    
    print("Welcome to the Enhanced Deep Researcher Agent.")
    print("I can provide concise research summaries based on my local knowledge base.")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nEnter your query: ")
        if user_query.lower() == 'exit':
            break

        print("\nProcessing your query...")
        response = handle_query(user_query, vectorizer, corpus_embeddings, corpus)
        
        print(f"\n**Research Summary:**\n{response}\n")

        # Interactive query refinement
        follow_up = input("Would you like to ask a follow-up question? (yes/no): ")
        if follow_up.lower() != 'yes':
            break

if __name__ == "__main__":
    main()