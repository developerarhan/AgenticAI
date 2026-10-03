import uuid
from django.db import models
from pgvector.django import VectorField, HnswIndex


class Document(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to="documents/")
    uploaded_at = models.DateTimeField(auto_now_add=True)


class DocumentChunk(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    content = models.TextField()
    chunk_index = models.IntegerField()
    page_number = models.IntegerField()
    # sentence-transformers/all-MiniLM-L6-v2
    embedding = VectorField(dimensions=384)


    class Meta:
        indexes = [
            # Create an HNSW index for sub-millisecond similarity search
            HnswIndex(
                name="embedding_hnsw_idx",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
        ]
        