import json
from backend.search_assessments import find_assessments


# -------------------- METRIC FUNCTIONS --------------------

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for url in retrieved_k if url in relevant_set)
    return hits / k if k else 0


def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for url in retrieved_k if url in relevant_set)
    return hits / len(relevant_set) if relevant_set else 0


def hit_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    return any(url in relevant for url in retrieved_k)


# -------------------- EVALUATION PIPELINE --------------------

def evaluate_model(labelled_data, k=5):

    precision_scores = []
    recall_scores = []
    hit_scores = []

    for row in labelled_data:
        query = row["query"]
        relevant_urls = row["assessment_urls"]

        results = find_assessments(query, k=k)
        retrieved_urls = [r["product_url"] for r in results]

        precision_scores.append(
            precision_at_k(retrieved_urls, relevant_urls, k)
        )
        recall_scores.append(
            recall_at_k(retrieved_urls, relevant_urls, k)
        )
        hit_scores.append(
            hit_at_k(retrieved_urls, relevant_urls, k)
        )

    return {
        "Precision@{}".format(k): round(sum(precision_scores) / len(precision_scores), 4),
        "Recall@{}".format(k): round(sum(recall_scores) / len(recall_scores), 4),
        "Hit@{}".format(k): round(sum(hit_scores) / len(hit_scores), 4)
    }


# -------------------- RUN EVALUATION --------------------

if __name__ == "__main__":

    with open("grouped_assessments.json", "r", encoding="utf-8") as f:
        labelled_data = json.load(f)

    metrics = evaluate_model(labelled_data, k=5)

    print("\nEvaluation Results")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
