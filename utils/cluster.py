from typing import List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster_by_length(strings: List[str]) -> List[str]:
    if not strings:
        return []

    lengths = np.array([[len(s)] for s in strings])  # Shape (n_samples, 1)
    unique_len_count = len(set(len(s) for s in strings))

    # If not enough samples or unique lengths, just return the original list
    if len(strings) < 2 or unique_len_count < 2:
        return strings

    # Avoid k > unique length values
    max_k = min(10, len(strings) - 1, unique_len_count)

    # Determine best k using silhouette score
    best_k = 2
    best_score = -1
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(lengths)
        score = silhouette_score(lengths, labels)
        if score > best_score:
            best_k = k
            best_score = score

    # Final clustering with best_k
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = final_kmeans.fit_predict(lengths)

    # Find cluster with smallest average length
    min_avg_length = float("inf")
    min_cluster_id = None
    for cluster_id in set(labels):
        cluster_lengths = lengths[labels == cluster_id]
        avg_len = np.mean(cluster_lengths)
        if avg_len < min_avg_length:
            min_avg_length = avg_len
            min_cluster_id = cluster_id

    return [s for s, label in zip(strings, labels) if label == min_cluster_id]



if __name__ == "__main__":
    test_strings = [
        "Short answer.",
        "A bit longer answer here.",
        "This is a significantly longer answer that goes into more detail.",
        "Tiny.",
        "Moderate length respasddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddaaaaaaaaaaaaaaaanse.",
        "An extremely long answer that is meant to test the clustering algosdaaaaaaaaaaaaaaaaarithm by providing a lot of information and context, making it the longest string in this particular test case.",
    ]
    clustered = cluster_by_length(test_strings)
    print("Clustered strings with shortest average length:")
    for s in clustered:
        print(f"- {s}")