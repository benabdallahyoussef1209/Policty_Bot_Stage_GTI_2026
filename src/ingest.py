# ============================================================
# src/ingest.py
# Regroupe tout ce qui concerne l'ingestion des documents :
# chargement -> découpage (chunking) -> embeddings -> stockage Chroma
# Basé sur check_documents.py, test_chunking.py et la partie "jour 4"
# de rag_chain_day5.py, réunis ici en un seul endroit réutilisable.
# ============================================================

import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Paramètres du projet, centralisés ici pour ne pas les répéter partout ---
DATA_DIR = "data/raw"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "policybot"

# Choix basé sur ton propre test (test_chunking.py) :
# chunk_size=1000 s'est montré plus clair, surtout pour les documents longs
# (rapports Apple / Tesla), tout en restant correct pour les documents courts.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def load_documents(dossier: str = DATA_DIR):
    """
    Charge tous les fichiers .txt et .pdf d'un dossier.
    Reprend exactement la logique de check_documents.py,
    mais sous forme de fonction réutilisable.
    """
    tous_les_docs = []
    fichiers = sorted(os.listdir(dossier))

    for nom_fichier in fichiers:
        chemin = os.path.join(dossier, nom_fichier)
        if nom_fichier.endswith(".txt"):
            loader = TextLoader(chemin, encoding="utf-8")
        elif nom_fichier.endswith(".pdf"):
            loader = PyPDFLoader(chemin)
        else:
            continue
        tous_les_docs.extend(loader.load())

    return tous_les_docs


def build_vectorstore():
    """
    Construit (ou recharge si elle existe déjà) la base vectorielle Chroma.
    Vérifie d'abord si la base contient déjà des documents avant d'en
    ajouter de nouveaux, pour éviter les doublons lors de relances multiples.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    nb_chunks_existants = vector_store._collection.count()

    if nb_chunks_existants == 0:
        print("Base vectorielle vide -> ingestion des documents en cours...")
        docs = load_documents()
        print(f"{len(docs)} document(s) chargé(s) depuis {DATA_DIR}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
        )
        all_splits = text_splitter.split_documents(docs)
        # Affiche le nombre de chunks générés par document source
        from collections import Counter
        sources = [chunk.metadata.get("source", "inconnu") for chunk in all_splits]
        compteur = Counter(sources)
        print("\nRépartition des chunks par document :")
        for source, nombre in sorted(compteur.items()):
            print(f"  {source} : {nombre} chunks")
        print()
        vector_store.add_documents(documents=all_splits)
        print(f"{len(all_splits)} chunks ajoutés à Chroma (collection '{COLLECTION_NAME}')")
    else:
        print(f"Base vectorielle déjà remplie : {nb_chunks_existants} chunks trouvés, aucun ajout.")

    return vector_store


def reset_vectorstore():
    """
    Utilitaire pour repartir de zéro si la base contient des doublons
    ou si tu changes ta stratégie de chunking.
    Supprime tous les documents de la collection, sans supprimer le dossier.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    ids_existants = vector_store.get()["ids"]
    if ids_existants:
        vector_store.delete(ids=ids_existants)
        print(f"{len(ids_existants)} chunks supprimés. La base est maintenant vide.")
    else:
        print("La base était déjà vide.")


if __name__ == "__main__":
    # Permet de tester l'ingestion seule avec : python src/ingest.py
    vs = build_vectorstore()
    print(f"\nNombre total de chunks dans la base : {vs._collection.count()}")