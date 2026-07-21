# ============================================================
# src/chain.py
# Regroupe tout ce qui concerne la génération de réponse :
# retriever -> prompt -> LLM -> réponse + sources utilisées
# ============================================================

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Paramètres du modèle, centralisés ici ---
MODEL_NAME = "qwen2.5:3b"
TEMPERATURE = 0  # réponses déterministes, importantes pour un système RAG
DEFAULT_K = 15   # nombre de chunks récupérés par question
SEED = 42          # pour la reproductibilité des tests
RAG_PROMPT = """
Réponds à la question en te basant uniquement sur le contexte ci-dessous.
Si l'information demandée n'est pas présente dans le contexte, réponds exactement :
"Je ne sais pas — cette information n'est pas présente dans les documents fournis."

Contexte :
{context}

Question :
{question}

Réponse :
"""

def format_docs(docs):
    """Fusionne le contenu textuel de plusieurs chunks en un seul bloc de contexte."""
    return "\n\n".join(doc.page_content for doc in docs)


def ask(question: str, vector_store, k: int = DEFAULT_K):
    """
    Pose une question au système RAG et renvoie un tuple (réponse, sources).
    Le retriever est appelé une seule fois, et son résultat (les chunks) sert
    à la fois à construire le contexte ET à extraire les sources.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # 1. Récupération des chunks pertinents pour la question
    docs = retriever.invoke(question)

    if not docs:
        return "Je ne sais pas — aucun document pertinent trouvé.", []

    # 2. Construction du contexte texte à partir des chunks récupérés
    context = format_docs(docs)

    # 3. Construction du prompt et appel au LLM local via Ollama
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    llm = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE, seed=SEED)
    chain = prompt | llm | StrOutputParser()

    reponse = chain.invoke({"context": context, "question": question})

    # 4. Extraction des sources uniques utilisées (sans doublons, triées)
    sources = sorted(set(doc.metadata.get("source", "inconnu") for doc in docs))

    return reponse, sources


if __name__ == "__main__":
    # Permet de tester la chaîne seule avec : python src/chain.py
    # (nécessite que la base vectorielle existe déjà - lance d'abord src/ingest.py)
    from ingest import build_vectorstore

    vector_store = build_vectorstore()
    question_test = "Quels sont les principaux risques mentionnés dans le rapport ?"
    reponse, sources = ask(question_test, vector_store)

    print(f"Question : {question_test}")
    print(f"Réponse : {reponse}")
    print(f"Sources : {sources}")