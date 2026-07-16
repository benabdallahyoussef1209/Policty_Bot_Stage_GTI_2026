# Importe le module os pour manipuler les fichiers et les dossiers
import os

# Importe les classes permettant de charger des fichiers texte et PDF
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Importe l'outil de découpage en chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================================
# Fonction : Charger tous les documents d'un dossier
# ==========================================================
def charger_documents(dossier):
    """Charge tous les fichiers .txt et .pdf d'un dossier."""

    documents = []

    for nom_fichier in sorted(os.listdir(dossier)):
        chemin = os.path.join(dossier, nom_fichier)

        if nom_fichier.endswith(".txt"):
            loader = TextLoader(chemin, encoding="utf-8")

        elif nom_fichier.endswith(".pdf"):
            loader = PyPDFLoader(chemin)

        else:
            continue

        documents.extend(loader.load())

    return documents


# ==========================================================
# Fonction : Découper les documents
# ==========================================================
def creer_chunks(documents, chunk_size, chunk_overlap):
    """Découpe les documents selon les paramètres donnés."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return splitter.split_documents(documents)


# ==========================================================
# Fonction : Afficher le premier chunk
# ==========================================================
def afficher_premier_chunk(chunks, taille):
    """Affiche le premier chunk généré."""

    print(f"=== Exemple avec chunk_size={taille} ===")
    print(chunks[0].page_content)
    print(f"\n(longueur : {len(chunks[0].page_content)} caractères)\n")


# ==========================================================
# Fonction : Afficher un chunk provenant d'un document précis
# ==========================================================
def afficher_chunk(chunks, mot_cle, titre):
    """Recherche puis affiche un chunk correspondant au mot-clé."""

    chunk = next(
        c for c in chunks
        if mot_cle.lower() in c.metadata.get("source", "").lower()
    )

    print(titre)
    print(chunk.page_content)
    print()


# ==========================================================
# Programme principal
# ==========================================================

# Dossier contenant les documents
dossier = "data/raw"

# Chargement des documents
tous_les_docs = charger_documents(dossier)

# Création des deux jeux de chunks
chunks_500 = creer_chunks(tous_les_docs, chunk_size=500, chunk_overlap=50)
chunks_1000 = creer_chunks(tous_les_docs, chunk_size=1000, chunk_overlap=100)

# Affichage des statistiques
print(f"Nombre total de documents chargés : {len(tous_les_docs)}")
print(f"Avec chunk_size=500  : {len(chunks_500)} chunks générés")
print(f"Avec chunk_size=1000 : {len(chunks_1000)} chunks générés\n")

# Premier chunk
afficher_premier_chunk(chunks_500, 500)
afficher_premier_chunk(chunks_1000, 1000)

# Chunks Tesla
afficher_chunk(
    chunks_500,
    "tesla",
    "=== Chunk Tesla (document long) avec chunk_size=500 ===",
)

afficher_chunk(
    chunks_1000,
    "tesla",
    "=== Chunk Tesla (document long) avec chunk_size=1000 ===",
)

# Chunks Réclamation
afficher_chunk(
    chunks_500,
    "reclamation1.txt",
    "=== Chunk réclamation (document court) avec chunk_size=500 ===",
)

afficher_chunk(
    chunks_1000,
    "reclamation1.txt",
    "=== Chunk réclamation (document court) avec chunk_size=1000 ===",
)