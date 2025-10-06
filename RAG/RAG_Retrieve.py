from pyarrow import large_list
from sentence_transformers import SentenceTransformer

from DB.Chroma import get_collection


def retrieve_from_rag(chroma_client):
    COLLECTION_SMALL = "wiki_small_chunks"
    COLLECTION_LARGE = "wiki_large_chunks"
    EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    query_text = "Who was the first person to use the printing press in Italy?"
    print(f"\n--- DEMO: Querying Small Chunks for: '{query_text}' ---")

    # Encode the query text using the same model
    query_embedding = custom_embed_function(embedding_model=embedding_model, texts=[query_text])

    small_collection = get_collection(
        client=chroma_client,
        collection_name=COLLECTION_SMALL
    )

    large_collection = get_collection(
        client=chroma_client,
        collection_name=COLLECTION_SMALL
    )

    # Search the small collection
    retrieved_small_chunks = small_collection.query(
        query_embeddings=query_embedding,
        n_results=1,
        include=['metadatas']
    )

    # Extract the parent ID of the most relevant small chunk
    parent_id = retrieved_small_chunks['metadatas'][0][0]['parent_id']
    print(f"Retrieved Small Chunk Parent ID: {parent_id}")

    full_context_result = large_collection.get(
        ids=[parent_id],
        include=['documents']
    )

    full_context = full_context_result['documents']
    print(full_context)
    #
    # print(f"\n--- FINAL CONTEXT FOR LLM (Large Chunk) ---")
    # print(f"Length of Context: {len(full_context)} characters.")
    # print("--- START CONTEXT ---")
    # print(full_context[:500] + "...")
    # print("--- END CONTEXT ---")

def custom_embed_function(texts, embedding_model):
    return embedding_model.encode(texts, convert_to_numpy=True).tolist()
