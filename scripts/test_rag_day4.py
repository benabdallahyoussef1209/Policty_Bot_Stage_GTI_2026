# ==========================================================
# RAG DAY 4 : Chargement, découpage et stockage vectoriel
# ==========================================================


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma



# ==========================================================
# Configuration
# ==========================================================

DOCUMENT_PATH = "data/exemple_finance.txt"

COLLECTION_NAME = "policybot_test"

CHROMA_PATH = "./chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"



# ==========================================================
# Fonction : Charger le document
# ==========================================================

def load_document(path):
    """
    Charge un fichier texte et retourne
    la liste des documents.
    """

    loader = TextLoader(
        path,
        encoding="utf-8"
    )

    documents = loader.load()


    print(
        f"Nombre de documents chargés : {len(documents)}"
    )


    print(
        f"Aperçu : {documents[0].page_content[:200]}"
    )


    return documents



# ==========================================================
# Fonction : Découper en chunks
# ==========================================================

def split_documents(documents):
    """
    Découpe les documents en morceaux
    avec chevauchement.
    """

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=500,

        chunk_overlap=50,

        add_start_index=True,
    )


    chunks = splitter.split_documents(
        documents
    )


    print(
        f"Document découpé en {len(chunks)} chunks"
    )


    return chunks



# ==========================================================
# Fonction : Créer la base vectorielle Chroma
# ==========================================================

def create_vectorstore(chunks):
    """
    Génère les embeddings et stocke
    les chunks dans Chroma.
    """


    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


    vector_store = Chroma(

        collection_name=COLLECTION_NAME,

        embedding_function=embeddings,

        persist_directory=CHROMA_PATH,

    )


    ids = vector_store.add_documents(
        documents=chunks
    )


    print(
        f"{len(ids)} chunks stockés dans Chroma"
    )


    return vector_store



# ==========================================================
# Programme principal
# ==========================================================

def main():


    # 1) Chargement du document
    docs = load_document(
        DOCUMENT_PATH
    )


    # 2) Découpage en chunks
    chunks = split_documents(
        docs
    )


    # 3) Embeddings + Chroma
    vector_store = create_vectorstore(
        chunks
    )


    return vector_store



# Point d'entrée
if __name__ == "__main__":

    vector_store = main()