# =========================
# Jour 4 : Préparation des données
# =========================

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Chargement du document
loader = TextLoader("data/politique_gti.txt", encoding="utf-8")
docs = loader.load()

# Découpage en chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
)

all_splits = text_splitter.split_documents(docs)

# Création des embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Création de la base vectorielle Chroma
vector_store = Chroma(
    collection_name="policybot_gti",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# Stockage des chunks
vector_store.add_documents(all_splits)

print(f"{len(all_splits)} chunks stockés dans Chroma")

# =========================
# Jour 5 : Chaîne RAG
# =========================

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Création du retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# Prompt
prompt = ChatPromptTemplate.from_template("""
Réponds à la question en te basant uniquement sur le contexte ci-dessous.
Si tu ne trouves pas la réponse dans le contexte, réponds : "Je ne sais pas".

Contexte :
{context}

Question :
{question}

Réponse :
""")

# Initialisation du LLM local
llm = ChatOllama(model="phi3", temperature=0)

# Fonction de formatage des documents récupérés
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Construction de la chaîne RAG
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Question utilisateur
question = input("Posez votre question : ")

# Génération de la réponse
reponse = rag_chain.invoke(question)

print(f"\nQuestion : {question}")
print(f"Réponse : {reponse}")