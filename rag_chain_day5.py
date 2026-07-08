#### code de jour 4+5####
#### jour 4 : chargement du document, découpage en chunks, création des embeddings et stockage dans Chroma  
#### CHARGDEMENTTT DE FICHIER DE TEST POUR RAG DAY 4###

from langchain_community.document_loaders import TextLoader  # Importe la classe permettant de charger un fichier texte.

loader = TextLoader("data/politique_gti.txt", encoding="utf-8")  # Crée un chargeur pour le fichier texte en UTF-8.
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
    collection_name="policybot_gti",  # Donne le nom "policybot_gti" à la collection qui contiendra les embeddings.
    embedding_function=embeddings,  # Indique à Chroma d'utiliser le modèle d'embeddings défini précédemment pour convertir les textes en vecteurs.
    persist_directory="./chroma_db",  # Spécifie le dossier où la base Chroma sera enregistrée sur le disque afin de pouvoir être réutilisée plus tard.
)

document_ids = vector_store.add_documents(  # Ajoute tous les chunks de texte à la base vectorielle.
    documents=all_splits  # Les documents découpés sont convertis en embeddings puis stockés dans Chroma avec leurs métadonnées.
)

print(f"{len(document_ids)} chunks stockés dans Chroma")  # Affiche le nombre de chunks qui ont été indexés et enregistrés dans la base vectorielle.

####-----------------------####
#jour 5 : création de la chaîne RAG complète avec LCEL (LangChain Expression Language)
# Importation du modèle Ollama permettant d'utiliser un LLM local (ex: Llama 3.2)
from langchain_ollama import ChatOllama

# Importation de l'outil permettant de créer un modèle de prompt avec des variables dynamiques
from langchain_core.prompts import ChatPromptTemplate

# Importation du parser permettant de récupérer uniquement le texte final généré par le LLM
from langchain_core.output_parsers import StrOutputParser

# Importation d'un composant permettant de transmettre une donnée sans modification dans la chaîne
from langchain_core.runnables import RunnablePassthrough


# Transformation du vector_store en retriever (moteur de recherche sémantique)
# k=5 signifie que les 5 documents/chunks les plus similaires seront récupérés
retriever = vector_store.as_retriever(search_kwargs={"k": 10})


# Création du template du prompt envoyé au modèle de langage
# {context} sera remplacé par les documents récupérés
# {question} sera remplacé par la question de l'utilisateur
prompt = ChatPromptTemplate.from_template("""
Réponds à la question en te basant uniquement sur le contexte ci-dessous.
Si tu ne trouves pas la réponse dans le contexte, dis "Je ne sais pas".

Contexte :
{context}

Question :
{question}

Réponse :
""")


# Initialisation du modèle LLM local Llama 3.2 avec Ollama
# Ce modèle sera chargé de générer la réponse à partir du contexte fourni
llm = ChatOllama(model="llama3.2:1b", temperature=0)


# Fonction permettant de transformer les documents récupérés en un seul texte
# Elle récupère uniquement le contenu de chaque chunk (page_content)
# puis les fusionne avec des retours à la ligne
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Création de la chaîne RAG complète avec LCEL (LangChain Expression Language)
rag_chain = (

    # Première étape :
    # - retriever récupère les documents pertinents depuis la base vectorielle
    # - format_docs transforme ces documents en un contexte texte
    # - RunnablePassthrough conserve la question originale
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }

    # Envoi du contexte et de la question dans le prompt
    | prompt

    # Envoi du prompt construit vers le modèle LLM pour générer une réponse
    | llm

    # Conversion de la réponse du LLM en simple texte
    | StrOutputParser()
)


# Question posée au système RAG concernant le contenu du document
question = "Combien de jours de télétravail par semaine sont autorisés ?"
# Exécution de toute la chaîne RAG :
# recherche des documents → création du contexte → génération de la réponse
reponse = rag_chain.invoke(question)


# Affichage de la question utilisateur
print(f"Question : {question}")

# Affichage de la réponse générée par le modèle
print(f"Réponse : {reponse}")
question = "Combien de jours de télétravail par semaine sont autorisés ?"
docs_trouves = retriever.invoke(question)
# ===== Test — Affichage des chunks récupérés par le retriever =====    
question = "Combien de jours de télétravail par semaine sont autorisés ?"
docs_trouves = retriever.invoke(question)

print("=== Chunks récupérés ===")
for i, doc in enumerate(docs_trouves):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content)
    print()

reponse = rag_chain.invoke(question)
print(f"Réponse : {reponse}")