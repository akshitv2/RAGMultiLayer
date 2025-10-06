# import uuid
#
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# from datasets import load_dataset
#
# from DB.Chroma import create_collection, create_collectionx
#
#
# def get_splitter(use_large: bool = False):
#     if use_large:
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1500,
#             chunk_overlap=150,
#             separators=["\n\n", "\n", " ", ""]
#         )
#     else:
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=400,
#             chunk_overlap=50,
#             separators=["\n\n", "\n", " ", ""]
#         )
#     return splitter
#
#
# def rag(config_yaml, chroma_client):
#     dataset = load_dataset(path="wikimedia/wikipedia", name="20231101.en", cache_dir=config_yaml["dataset"]["dir"],
#                            split="train[:5]")
#     max_articles = min(len(dataset), config_yaml["dataset"]["max_articles"])
#     dataset = dataset.select(range(max_articles))
#
#     EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
#     COLLECTION_SMALL = "wiki_small_chunks"
#     COLLECTION_LARGE = "wiki_large_chunks"
#
#     # Initialize the Embedding Model
#
#
#     documents_to_process = [doc['text'] for doc in dataset]
#     small_chunks_data = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
#     large_chunks_data = {"ids": [], "documents": [], "metadatas": []}
#
#     large_splitter = get_splitter(use_large=True)
#     small_splitter = get_splitter(use_large=False)
#
#     small_collection = create_collectionx(chroma_client, COLLECTION_SMALL)
#     large_collection = create_collectionx(chroma_client, COLLECTION_LARGE)
#
#     # Process each document
#     for doc_index, document in enumerate(documents_to_process):
#         large_chunks = large_splitter.create_documents([document])
#         parent_map = {}
#         for l_chunk in large_chunks:
#             parent_id = str(uuid.uuid4())
#             large_chunks_data["ids"].append(parent_id)
#             large_chunks_data["documents"].append(l_chunk.page_content)
#
#             # l_chunk.metadata["doc_index"] = doc_index
#             # Optional Metadata, might be needed in case one wants better traceability to source
#
#             # l_chunk.metadata["parent_id"] = parent_id
#             # large_chunks_data["metadatas"].append(l_chunk.metadata)
#             # todo: Integrate metadata to refine RAG
#             # parent_map[l_chunk.page_content] = parent_id  # Temporary map
#             small_chunks = small_splitter.create_documents([l_chunk.page_content])
#
#             # Next, collect all small chunk data and map them to the nearest parent ID
#             for s_chunk in small_chunks:
#                 child_id = str(uuid.uuid4())
#
#                 s_chunk.metadata["parent_id"] = parent_id
#
#                 # s_chunk.metadata["doc_index"] = doc_index
#                 # Optional Metadata, might be needed in case one wants better traceability to source
#
#                 # Generate embedding for the small chunk immediately
#                 # embedding = custom_embed_function(embedding_model = embedding_model, texts = [s_chunk.page_content])[0]
#                 small_chunks_data["ids"].append(child_id)
#
#                 small_chunks_data["documents"].append(s_chunk.page_content)
#                 # small_chunks_data["metadatas"].append(s_chunk.metadata)
#                 # todo: Integrate metadata to refine RAG
#                 # small_chunks_data["embeddings"].append(embedding)
#
#                 small_collection.add(
#                     ids=[child_id],
#                     documents=[s_chunk.page_content],
#                     metadatas=[s_chunk.metadata],
#                     # embeddings=small_chunks_data["embeddings"]  # Insert the pre-computed embeddings
#                 )
#
#     # print(f"\nInserting {len(large_chunks_data['ids'])} large chunks (parents) into '{COLLECTION_LARGE}'...")
#     # large_collection.add(
#     #     ids=large_chunks_data["ids"],
#     #     documents=large_chunks_data["documents"],
#     #     metadatas=large_chunks_data["metadatas"],
#     # )
#     # print(f"Successfully added large chunks. Collection count: {large_collection.count()}")
#
#
# # def retrieve_from_rag():
# #     query_text = "Who was the first person to use the printing press in Italy?"
# #     print(f"\n--- DEMO: Querying Small Chunks for: '{query_text}' ---")
# #
# #     # Encode the query text using the same model
# #     query_embedding = custom_embed_function([query_text])
# #
# #     # Search the small collection
# #     retrieved_small_chunks = small_collection.query(
# #         query_embeddings=query_embedding,
# #         n_results=1,
# #         include=['metadatas']  # We only need the parent_id from metadata
# #     )
# #
# #     # Extract the parent ID of the most relevant small chunk
# #     parent_id = retrieved_small_chunks['metadatas'][0][0]['parent_id']
# #     print(f"Retrieved Small Chunk Parent ID: {parent_id}")
# #
# #     # 2. Retrieve the full context (Large Chunk)
# #     # Use the Parent ID to get the full, original context from the large collection
# #     full_context_result = large_collection.get(
# #         ids=[parent_id],
# #         include=['documents']
# #     )
# #
# #     full_context = full_context_result['documents'][0]
# #
# #     print(f"\n--- FINAL CONTEXT FOR LLM (Large Chunk) ---")
# #     print(f"Length of Context: {len(full_context)} characters.")
# #     print("--- START CONTEXT ---")
# #     print(full_context[:500] + "...")
# #     print("--- END CONTEXT ---")
