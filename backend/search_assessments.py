import os
import requests
import numpy as np
import faiss
import pickle
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
}

# -------------------- LOAD RESOURCES --------------------

# Load cleaned product data
products_df = pd.read_csv("products_cleaned.csv")

# Load FAISS index
index = faiss.read_index("vector.index")

# Load vector-to-text mapping
with open("vector_texts.pkl", "rb") as f:
    vector_texts = pickle.load(f)

# Load SAME embedding model used during indexing
# model = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------- CORE FUNCTIONS --------------------

def embed_query(query: str):
    """Embed user query using SentenceTransformer"""
    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": query}
    )

    response.raise_for_status()

    embedding = response.json()

    # HF returns [768] or [[768]]
    if isinstance(embedding[0], list):
        embedding = embedding[0]

    return np.array([embedding], dtype="float32")


def find_assessments(user_query, k=5):
    """
    Perform semantic search over SHL assessments
    """

    query_vector = embed_query(user_query)

    # FAISS similarity search
    distances, indices = index.search(query_vector, k)

    results = []

    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue

        product = vector_texts[idx]  # this should be a dict

        result = {
            "product_name": str(product["product_name"]),
            "product_url": str(product["product_url"]),
            "test_type": product["test_type"],  # already list / string
            "product_description": str(product["product_description"]),
            "remote_testing_available": str(product["remote_testing_available"]),
            "Job_levels": product["Job_levels"],
            "Score": float(round(1 / (1 + float(dist)), 4))
        }

        results.append(result)


    return results


def print_assessments(user_query, k=5):
    """
    Print top-k matching assessments in a readable format
    """

    results = find_assessments(user_query, k=k)

    print(f"\nTop {len(results)} Matching SHL Assessments\n")

    for i, res in enumerate(results, 1):
        print(f"{i}. {res['product_name']}")
        print(f"   Test Type: {res['test_type']}")
        print(f"   Job Levels: {res['Job_levels']}")
        print(f"   Remote Testing: {res['remote_testing_available']}")
        print(f"   Score: {res['Score']}")
        print(f"   URL: {res['product_url']}")
        print("-" * 80)

    return results


# -------------------- CLI TEST --------------------

if __name__ == "__main__":
    query = "Entry-level accounting assessment with remote testing"
    print_assessments(query, k=5)
