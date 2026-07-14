# Importation du module os pour manipuler les fichiers et les dossiers
import os

# Importation des chargeurs de documents de LangChain
# TextLoader : permet de lire les fichiers texte (.txt)
# PyPDFLoader : permet de lire les fichiers PDF (.pdf)
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Définition du dossier contenant les documents à traiter
dossier = "data/raw"

# Récupération de la liste des fichiers présents dans le dossier
# sorted() permet de les trier par ordre alphabétique
fichiers = sorted(os.listdir(dossier))

# Affichage du nombre total de fichiers détectés
print(f"Nombre de fichiers trouvés : {len(fichiers)}\n")

# Parcours de chaque fichier contenu dans le dossier
for nom_fichier in fichiers:

    # Construction du chemin complet vers le fichier
    chemin = os.path.join(dossier, nom_fichier)

    # Vérification si le fichier est un fichier texte
    if nom_fichier.endswith(".txt"):

        # Création d'un chargeur adapté aux fichiers texte avec l'encodage UTF-8
        loader = TextLoader(chemin, encoding="utf-8")

    # Vérification si le fichier est un document PDF
    elif nom_fichier.endswith(".pdf"):

        # Création d'un chargeur spécialisé pour les fichiers PDF
        loader = PyPDFLoader(chemin)

    # Si le format du fichier n'est pas pris en charge, on passe au fichier suivant
    else:
        continue

    # Chargement du contenu du document dans une liste d'objets Document
    docs = loader.load()

    # Affichage du nom du fichier en cours de traitement
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