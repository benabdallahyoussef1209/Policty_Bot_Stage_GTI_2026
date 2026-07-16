# Importation du module os pour manipuler les fichiers et les dossiers
import os

# Importation des chargeurs de documents de LangChain
from langchain_community.document_loaders import TextLoader, PyPDFLoader


# ==========================================================
# Fonction : Charger un document selon son extension
# ==========================================================
def charger_document(chemin):
    """
    Charge un fichier texte (.txt) ou PDF (.pdf)
    et retourne une liste d'objets Document.
    """

    if chemin.endswith(".txt"):
        loader = TextLoader(chemin, encoding="utf-8")

    elif chemin.endswith(".pdf"):
        loader = PyPDFLoader(chemin)

    else:
        return None

    return loader.load()


# ==========================================================
# Programme principal
# ==========================================================

# Dossier contenant les documents
dossier = "data/raw"

# Liste des fichiers du dossier
fichiers = sorted(os.listdir(dossier))

# Affiche le nombre total de fichiers
print(f"Nombre de fichiers trouvés : {len(fichiers)}\n")

# Parcourt chaque fichier
for nom_fichier in fichiers:

    # Construit le chemin complet du fichier
    chemin = os.path.join(dossier, nom_fichier)

    # Charge le document
    docs = charger_document(chemin)

    # Ignore les formats non pris en charge
    if docs is None:
        continue

    # Affiche le nom du document
    print(f"=== {nom_fichier} ===")
    # Affichage du nombre de blocs obtenus
    # Pour un PDF, cela correspond généralement au nombre de pages
    # Pour un fichier texte, cela correspond généralement à un seul bloc
    print(f"Nombre de sections/pages : {len(docs)}")

    # Affichage de la taille du premier bloc en nombre de caractères
    print(f"Longueur du 1er bloc : {len(docs[0].page_content)} caractères")

    # Affichage d'un aperçu du contenu du document
    print("Aperçu (200 premiers caractères) :")

    # Affichage des 200 premiers caractères du premier bloc
    print(docs[0].page_content[:200])

    # Ligne vide pour améliorer la lisibilité entre les fichiers
    print()