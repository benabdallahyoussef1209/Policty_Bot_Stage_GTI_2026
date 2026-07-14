# Importe le module os pour manipuler les fichiers et les dossiers du système.
import os

# Importe les classes permettant de charger des fichiers texte (.txt) et PDF (.pdf).
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Importe le découpeur de texte qui divise les documents en plusieurs morceaux (chunks).
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Importe le modèle d'embeddings Hugging Face qui transforme le texte en vecteurs numériques.
from langchain_huggingface import HuggingFaceEmbeddings

# Importe Chroma, la base de données vectorielle utilisée pour stocker les embeddings.
from langchain_chroma import Chroma

# Définit le chemin du dossier contenant tous les documents à indexer.
dossier = "data/raw"

# Récupère la liste de tous les fichiers du dossier et les trie par ordre alphabétique.
fichiers = sorted(os.listdir(dossier))

# ================= Chargement de tous les documents =================

# Crée une liste vide qui contiendra tous les documents chargés.
tous_les_docs = []

# Parcourt chaque fichier présent dans le dossier.
for nom_fichier in fichiers:

    # Construit le chemin complet du fichier.
    chemin = os.path.join(dossier, nom_fichier)

    # Vérifie si le fichier est un fichier texte (.txt).
    if nom_fichier.endswith(".txt"):

        # Crée un chargeur de fichier texte avec l'encodage UTF-8.
        loader = TextLoader(chemin, encoding="utf-8")

    # Vérifie si le fichier est un fichier PDF.
    elif nom_fichier.endswith(".pdf"):

        # Crée un chargeur de fichier PDF.
        loader = PyPDFLoader(chemin)

    # Ignore les fichiers qui ne sont ni des .txt ni des .pdf.
    else:
        continue

    # Charge le contenu du document et l'ajoute à la liste générale.
    tous_les_docs.extend(loader.load())

# Affiche le nombre total de documents chargés.
print(f"Nombre total de documents chargés : {len(tous_les_docs)}")
# ===== Découpage avec la config retenue au Jour 3 =====
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(tous_les_docs)

print(f"Document découpé en {len(all_splits)} chunks")
#============== nombre de chunks par document======#
from collections import Counter

# Compte les chunks par fichier source
sources = [chunk.metadata.get("source", "inconnu") for chunk in all_splits]
compteur = Counter(sources)

print("\n=== Répartition des chunks par document ===")
for source, nombre in sorted(compteur.items()):
    print(f"{source} : {nombre} chunks")
     # ================= Création des embeddings =================

# Charge le modèle d'embeddings all-MiniLM-L6-v2 de Sentence Transformers.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# ================= Création de la base vectorielle =================

# Crée (ou ouvre) une base Chroma.
vector_store = Chroma(

    # Nom de la collection qui contiendra les documents.
    collection_name="policybot_real_docs",

    # Modèle utilisé pour transformer les textes en vecteurs.
    embedding_function=embeddings,

    # Dossier dans lequel la base sera enregistrée.
    persist_directory="./chroma_db",
)

# Ajoute tous les chunks dans la base vectorielle.
document_ids = vector_store.add_documents(documents=all_splits)

# Affiche le nombre de chunks stockés dans Chroma.
print(f"{len(document_ids)} chunks stockés dans Chroma (collection: policybot_real_docs)")