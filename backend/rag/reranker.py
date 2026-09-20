from sentence_transformers import CrossEncoder


print("RERANKER MODULE LOADED")


# MS MARCO Cross-Encoder
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


# Load the model only once when this module is imported
_reranker = CrossEncoder(
    MODEL_NAME,
    max_length=512
)


def get_reranker():
    return _reranker


def rerank_documents(query, results, top_k=5):

    if not results:
        return []

    reranker = get_reranker()

    # -----------------------------------------
    # Extract documents
    # -----------------------------------------

    documents = [
        document
        for document, _ in results
    ]

    # -----------------------------------------
    # Create query-document pairs
    # -----------------------------------------

    pairs = [
        (
            query,
            document.page_content
        )
        for document in documents
    ]

    try:

        scores = reranker.predict(
            pairs,
            show_progress_bar=False
        )

    except Exception as e:

        print(
            f"RERANKER ERROR: {e}"
        )

        return []

    # -----------------------------------------
    # Combine documents with scores
    # -----------------------------------------

    reranked = []

    for document, score in zip(
        documents,
        scores
    ):

        reranked.append(
            (
                document,
                float(score)
            )
        )

    # -----------------------------------------
    # Highest relevance first
    # -----------------------------------------

    reranked.sort(
        key=lambda item: item[1],
        reverse=True
    )

    # -----------------------------------------
    # Return top documents
    # -----------------------------------------

    return reranked[:top_k]