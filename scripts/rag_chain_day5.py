# =========================
# Jour 4 : Préparation des données
# =========================

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ==========================================================
# Fonction : Construire la base vectorielle
# ==========================================================
def build_vectorstore():
    """
    Charge le document, le découpe en chunks,
    génère les embeddings et les stocke dans Chroma.
    """

    # Chargement du document
    loader = TextLoader("data/politique_gti.txt", encoding="utf-8")
    docs = loader.load()

    # Découpage
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
    )

    chunks = splitter.split_documents(docs)

    # Modèle d'embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Base vectorielle
    vector_store = Chroma(
        collection_name="policybot_gti",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

    # Stockage
    vector_store.add_documents(chunks)

    print(f"{len(chunks)} chunks stockés dans Chroma")

    return vector_store


# ==========================================================
# Fonction : Formater les documents récupérés
# ==========================================================
def format_docs(documents):
    """Assemble les chunks récupérés en un seul texte."""

    return "\n\n".join(doc.page_content for doc in documents)


# ==========================================================
# Fonction : Construire la chaîne RAG
# ==========================================================
def build_rag_chain(vector_store):
    """Construit la chaîne RAG."""

    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    prompt = ChatPromptTemplate.from_template(
        """
Réponds à la question en te basant uniquement sur le contexte ci-dessous.
Si tu ne trouves pas la réponse dans le contexte, réponds : "Je ne sais pas".

Contexte :
{context}

Question :
{question}

Réponse :
"""
    )

    llm = ChatOllama(
        model="phi3",
        temperature=0,
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ==========================================================
# Programme principal
# ==========================================================

# Construction de la base vectorielle
vector_store = build_vectorstore()

# Construction de la chaîne RAG
rag_chain = build_rag_chain(vector_store)

# Question utilisateur
question = input("Posez votre question : ")

# Génération de la réponse
reponse = rag_chain.invoke(question)

print(f"\nQuestion : {question}")
print(f"Réponse : {reponse}")