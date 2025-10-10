from sentence_transformers import SentenceTransformer

from DB.Chroma import get_collection

def inspect_db(chroma_client):
    LIMIT = 1000
    COLLECTION_SMALL = "wiki_small_chunks"
    COLLECTION_LARGE = "wiki_large_chunks"

    small_collection = get_collection(
        client=chroma_client,
        collection_name=COLLECTION_SMALL
    )
    large_collection = get_collection(
        client=chroma_client,
        collection_name=COLLECTION_LARGE
    )

    # --- 1. INSPECT SMALL CHUNKS (Search Index) ---
    print(f"--- First {LIMIT} Small Chunks (Search Index) ---")
    small_data = small_collection.get(limit=LIMIT, include=['documents', 'metadatas'])
    print(f"\n--- First {LIMIT} Large Chunks (Context Source) ---")
    large_data = large_collection.get(limit=LIMIT, include=['documents'])
    for i in range(len(small_data['ids'])):
        print(f"\nID: {small_data['ids'][i]}")
        print(f"Parent_ID: {small_data['metadatas'][i]['parent_id']}")
        print(f"Content: {small_data['documents'][i][:700]}...")

    for i in range(len(large_data['ids'])):
        print(f"\nID: {large_data['ids'][i]}")
        print(f"Content Length: {len(large_data['documents'][i])}")
        print(f"Content: {large_data['documents'][i][:100]}...")

    full_context_result = large_collection.get(
        ids=["67bbe5d8-d754-4680-89c7-09cbd8808814"],
        include=['documents']
    )
    print(full_context_result)

def retrieve_from_rag(chroma_client):
    COLLECTION_SMALL = "wiki_small_chunks"
    COLLECTION_LARGE = "wiki_large_chunks"
    EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    query_text = "Where is alabama?"
    print(f"\n--- DEMO: Querying Small Chunks for: '{query_text}' ---")

    # Encode the query text using the same model
    query_embedding = custom_embed_function(embedding_model=embedding_model, texts=[query_text])

    small_collection = get_collection(
        client=chroma_client,
        collection_name=COLLECTION_SMALL
    )

    large_collection = get_collection(
        client=chroma_client,
        collection_name=COLLECTION_LARGE
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

    # full_context = full_context_result['documents']
    print(full_context_result)
    #
    # print(f"\n--- FINAL CONTEXT FOR LLM (Large Chunk) ---")
    # print(f"Length of Context: {len(full_context)} characters.")
    # print("--- START CONTEXT ---")
    # print(full_context[:500] + "...")
    # print("--- END CONTEXT ---")

def custom_embed_function(texts, embedding_model):
    return embedding_model.encode(texts, convert_to_numpy=True).tolist()
