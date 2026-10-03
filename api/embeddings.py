from sentence_transformers import SentenceTransformer
from typing import List

model = SentenceTransformer("all-MiniLM-L6-v2") #384


def generate_embeddings(text_list: List[str]) -> List[List[float]]:
    """Generates vector embeddings for a list of strings."""
    document_embeddings = model.encode(text_list)
    return document_embeddings.tolist()