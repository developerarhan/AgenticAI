import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from pgvector.django import CosineDistance
from sentence_transformers import SentenceTransformer
from groq import Groq
from .models import Document, DocumentChunk
from .pipeline import extract_pdf_pages, recursive_chunk_text
from .embeddings import generate_embeddings



model = SentenceTransformer("all-MiniLM-L6-v2") #384

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class UploadDocumentView(APIView):
    def post(self, request):
        # print("\n========== DOCUMENT UPLOAD STARTED ==========")
        file_obj = request.FILES.get('file')
        if not file_obj:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        # print(f"File name: {file_obj.name}")
        # print(f"File size: {file_obj.size} bytes")
        # print(f"Content type: {file_obj.content_type}")

        # Save document
        doc = Document.objects.create(title=file_obj.name, file=file_obj)
        # print("\n--- Document Saved ---")
        # print(f"Document ID: {doc.id}")
        # print(f"Document title: {doc.title}")
        # print(f"Document path: {doc.file.path}")

        # Extract and chunk text
        # print("\n--- Extracting PDF Pages ---")
        pages = extract_pdf_pages(doc.file.path)
        # print(f"Total pages extracted: {len(pages)}")
        # print("\n--- Creating Chunks ---")
        raw_chunks = recursive_chunk_text(pages)
        # print(f"Total chunks created: {len(raw_chunks)}")

        # Batch Embeddings
        # print("\n--- Preparing Embeddings ---")
        chunk_texts = [c["content"] for c in raw_chunks]
        # print(f"Number of texts to embed: {len(chunk_texts)}")
        # print("\n--- Generating Embeddings ---")
        embeddings = generate_embeddings(chunk_texts)
        # print(f"Number of embeddings generated: {len(embeddings)}")

        # Bulk Insert Chunks
        chunk_objects = []
        for raw_chunk, vector in zip(raw_chunks, embeddings):
            chunk_objects.append(
                DocumentChunk(
                    document=doc,
                    content=raw_chunk["content"],
                    chunk_index=raw_chunk["chunk_index"],
                    page_number=raw_chunk["page_number"],
                    embedding=vector
                )
            )

        DocumentChunk.objects.bulk_create(chunk_objects)

        return Response({"document_id": doc.id, "total_chunks": len(chunk_objects)}, status=status.HTTP_201_CREATED)

    
class RAGQueryView(APIView):
    def post(self, request):
        query = request.data.get("query")
        document_id = request.data.get("document_id")
        top_k = request.data.get("top_k", 4)

        if not query:
            return Response({"error": "Query parameter missing"}, status=status.HTTP_400_BAD_REQUEST)

        # Embed Query
        # print(f"user_query: {query} and document_id: {document_id}")
        query_embedding = model.encode(query, normalize_embeddings=True).tolist()
        # print(f"query_embedding: {query_embedding}")

        # Top-K Similarity Search via pgvector
        qs = DocumentChunk.objects.all()
        if document_id:
            qs = qs.filter(document_id=document_id)

        # print("Total chunks:", qs.count())

        matched_chunks = qs.annotate(
            distance=CosineDistance("embedding", query_embedding)
        ).order_by("distance")[:top_k]

        # print("Matched chunks:", len(matched_chunks))

        # Construct Context with Strict Structural IDs
        context_blocks = []
        source_citations = []

        for idx, chunk in enumerate(matched_chunks):
            citation_id = f"SRC_{idx+1}"
            context_blocks.append(
                f"[{citation_id}] Page {chunk.page_number}:\n{chunk.content}"
            )
            source_citations.append({
                "citation_id": citation_id,
                "chunk_id": str(chunk.id),
                "page_number": chunk.page_number,
                "content": chunk.content,
                "similarity_score": round(1 - chunk.distance, 4)
            })

        context_str = "\n\n".join(context_blocks)

        # Generate Answer
        system_prompt = (
            "You are an academic Q&A assistant. Answer the user question based ONLY on the context below. "
            "For every statement you make, cite the source using the exact notation [SRC_X]. "
            "If the answer cannot be determined from the context, state that explicitly.\n\n"
            f"CONTEXT:\n{context_str}"
        )

        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.3
        )

        return Response({
            "answer": response.choices[0].message.content,
            "citations": source_citations
        })

