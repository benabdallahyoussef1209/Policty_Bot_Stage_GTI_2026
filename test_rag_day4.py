#### CHARGDEMENTTT DE FICHIER DE TEST POUR RAG DAY 4###

from langchain_community.document_loaders import TextLoader  # Importe la classe permettant de charger un fichier texte.

loader = TextLoader("data/exemple_finance.txt", encoding="utf-8")  # Crée un chargeur pour le fichier texte en UTF-8.
docs = loader.load()  # Charge le contenu du fichier dans une liste de documents.

print(f"Nombre de documents chargés : {len(docs)}")  # Affiche le nombre de documents chargés.
print(f"Aperçu : {docs[0].page_content[:200]}")  # Affiche les 200 premiers caractères du premier document.
### DECOUPAGE EN CHUNKSS###

from langchain_text_splitters import RecursiveCharacterTextSplitter  # Importe le découpeur de texte de LangChain.

text_splitter = RecursiveCharacterTextSplitter(  # Crée un objet pour découper les documents en morceaux.
    chunk_size=500,      # Définit une taille maximale de 500 caractères par morceau.
    chunk_overlap=50,    # Fait chevaucher les morceaux de 50 caractères pour conserver le contexte.
    add_start_index=True,  # Ajoute l'index de début de chaque morceau dans les métadonnées.
)

all_splits = text_splitter.split_documents(docs)  # Découpe les documents chargés en plusieurs morceaux.

print(f"Document découpé en {len(all_splits)} morceaux (chunks)")  # Affiche le nombre total de morceaux créés.

###  Embeddings + stockage dans Chroma ###
 
from langchain_huggingface import HuggingFaceEmbeddings  # Importe la classe permettant de transformer du texte en vecteurs (embeddings) avec un modèle Hugging Face.
from langchain_chroma import Chroma  # Importe Chroma, une base de données vectorielle utilisée pour stocker et rechercher les embeddings.

embeddings = HuggingFaceEmbeddings(  # Crée un objet chargé de générer les embeddings des textes.
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Utilise le modèle "all-MiniLM-L6-v2", rapide et performant pour convertir chaque chunk en un vecteur numérique.
)

vector_store = Chroma(  # Crée une base de données vectorielle Chroma.
    collection_name="policybot_test",  # Donne le nom "policybot_test" à la collection qui contiendra les embeddings.
    embedding_function=embeddings,  # Indique à Chroma d'utiliser le modèle d'embeddings défini précédemment pour convertir les textes en vecteurs.
    persist_directory="./chroma_db",  # Spécifie le dossier où la base Chroma sera enregistrée sur le disque afin de pouvoir être réutilisée plus tard.
)

document_ids = vector_store.add_documents(  # Ajoute tous les chunks de texte à la base vectorielle.
    documents=all_splits  # Les documents découpés sont convertis en embeddings puis stockés dans Chroma avec leurs métadonnées.
)

print(f"{len(document_ids)} chunks stockés dans Chroma")  # Affiche le nombre de chunks qui ont été indexés et enregistrés dans la base vectorielle.