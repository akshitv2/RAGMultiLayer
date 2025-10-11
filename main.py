import os

from qdrant_client import QdrantClient

from DB.qDrant import check_qdrant_status, delete_all_existing_collections, QdrantConfig
from RAG.RAG import rag_parallel
from config.loadConfig import load_project_config

if __name__ == "__main__":
    config = load_project_config(os.path.join(os.getcwd(), "config/config.yaml"))
    qdrantConfig = QdrantConfig(
            host=config["db"]["qdrant"]["connection"]["host"],
            port=config["db"]["qdrant"]["connection"]["port"],
            timeout=config["db"]["qdrant"]["connection"]["timeout"]
    )
    status, client = check_qdrant_status(qdrantConfig)
    if not status:
        print("Unable to connect to qdrant")
        exit()
    # data_populated = False
    if config['mode']['prepare_data']:
        delete_all_existing_collections(qdrantConfig)
    if config['mode']['prepare_data']:
        rag_parallel(config, client)

    # inspect_db(chroma_client)
    # retrieve_from_rag(chroma_client)
