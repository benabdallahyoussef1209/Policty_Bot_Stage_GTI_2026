# Importe le module os pour manipuler les fichiers et les dossiers
import os

# Importe les classes permettant de charger des fichiers texte et PDF
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Importe l'outil qui découpe les documents en plusieurs morceaux (chunks)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Définit le chemin du dossier contenant les documents à traiter
dossier = "data/raw"

# Récupère la liste des fichiers du dossier et les trie par ordre alphabétique
fichiers = sorted(os.listdir(dossier))

# Initialise une liste vide qui contiendra tous les documents chargés
tous_les_docs = []

# Parcourt tous les fichiers présents dans le dossier
for nom_fichier in fichiers:

    # Construit le chemin complet du fichier
    chemin = os.path.join(dossier, nom_fichier)

    # Vérifie si le fichier est un fichier texte
    if nom_fichier.endswith(".txt"):

        # Crée un chargeur pour les fichiers texte en UTF-8
        loader = TextLoader(chemin, encoding="utf-8")

    # Vérifie si le fichier est un fichier PDF
    elif nom_fichier.endswith(".pdf"):

        # Crée un chargeur pour les fichiers PDF
        loader = PyPDFLoader(chemin)

    # Ignore tous les autres types de fichiers
    else:
        continue

    # Charge le contenu du fichier et l'ajoute à la liste de tous les documents
    tous_les_docs.extend(loader.load())

# Crée un découpeur avec des morceaux de 500 caractères
splitter_500 = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Taille maximale d'un chunk
    chunk_overlap=50     # Chevauchement de 50 caractères entre deux chunks
)

# Découpe tous les documents selon cette configuration
chunks_500 = splitter_500.split_documents(tous_les_docs)

# Crée un deuxième découpeur avec des morceaux de 1000 caractères
splitter_1000 = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # Taille maximale d'un chunk
    chunk_overlap=100    # Chevauchement de 100 caractères
)

# Découpe tous les documents avec cette deuxième configuration
chunks_1000 = splitter_1000.split_documents(tous_les_docs)

# Affiche le nombre total de documents chargés
print(f"Nombre total de documents chargés : {len(tous_les_docs)}")

# Affiche le nombre de chunks obtenus avec une taille de 500 caractères
print(f"Avec chunk_size=500  : {len(chunks_500)} chunks générés")

# Affiche le nombre de chunks obtenus avec une taille de 1000 caractères
print(f"Avec chunk_size=1000 : {len(chunks_1000)} chunks générés")

# Affiche une ligne vide pour améliorer la lisibilité
print()

# Affiche un titre pour l'exemple avec des chunks de 500 caractères
print("=== Exemple avec chunk_size=500 ===")

# Affiche le contenu du premier chunk
print(chunks_500[0].page_content)

# Affiche la longueur du premier chunk en nombre de caractères
print(f"(longueur : {len(chunks_500[0].page_content)} caractères)")

# Affiche une ligne vide
print()

# Affiche un titre pour l'exemple avec des chunks de 1000 caractères
print("=== Exemple avec chunk_size=1000 ===")

# Affiche le contenu du premier chunk
print(chunks_1000[0].page_content)

# Affiche la longueur du premier chunk en nombre de caractères
print(f"(longueur : {len(chunks_1000[0].page_content)} caractères)")
## affichage pas seulement le premier document##
# Trouve un chunk qui vient de Tesla (document long)
chunk_tesla = next(c for c in chunks_500 if "tesla" in c.metadata.get("source", "").lower())
print("=== Chunk Tesla (document long) avec chunk_size=500 ===")
print(chunk_tesla.page_content)
print()

# Trouve un chunk qui vient d'une réclamation1 (document court)
chunk_reclamation = next(c for c in chunks_500 if "reclamation1.txt" in c.metadata.get("source", "").lower())
print("=== Chunk réclamation (document court) avec chunk_size=500 ===")
print(chunk_reclamation.page_content)
#de mm pour 1000 chunks#
chunk_tesla2 = next(c for c in chunks_1000 if "tesla" in c.metadata.get("source", "").lower())
print("=== Chunk Tesla (document long) avec chunk_size=1000 ===")
print(chunk_tesla.page_content)
print()

# Trouve un chunk qui vient d'une réclamation1 (document court)
chunk_reclamation2 = next(c for c in chunks_1000 if "reclamation1.txt" in c.metadata.get("source", "").lower())
print("=== Chunk réclamation (document court) avec chunk_size=1000 ===")
print(chunk_reclamation2.page_content)