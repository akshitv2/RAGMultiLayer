import os

from DB.Chroma import run_db
from RAG.RAG import rag
from RAG.RAG_Retrieve import retrieve_from_rag
from config.loadConfig import load_project_config

if __name__ == "__main__":
    chroma_client = run_db()
    config_yaml = load_project_config(os.path.join(os.getcwd(), "config/config.yaml"))

    # rag(config_yaml, chroma_client)
    retrieve_from_rag(chroma_client)
