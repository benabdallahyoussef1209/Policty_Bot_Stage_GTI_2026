# ==========================================================
# Test Retriever RAG - PolicyBot
# ==========================================================

import sys
import os


# Ajoute le dossier parent pour permettre l'import depuis src/
sys.path.append(
    os.path.join(os.path.dirname(__file__), "..")
)


# Importation de la fonction de création/chargement Chroma
from src.ingest import build_vectorstore



# ==========================================================
# Fonction : Créer le retriever
# ==========================================================

def create_retriever(vector_store, k=3):
    """
    Crée un retriever permettant de récupérer
    les k chunks les plus pertinents.
    """

    return vector_store.as_retriever(
        search_kwargs={"k": k}
    )



# ==========================================================
# Fonction : Afficher les documents récupérés
# ==========================================================

def display_documents(docs):
    """
    Affiche les chunks retournés par le retriever.
    """

    for i, doc in enumerate(docs, start=1):

        print(
            f"--- Chunk {i} "
            f"(source : {doc.metadata.get('source')}) ---"
        )

        print(
            doc.page_content[:200]
        )

        print()



# ==========================================================
# Fonction : Afficher les scores de similarité
# ==========================================================

def display_similarity_scores(vector_store, question, k=3):
    """
    Recherche les documents avec leurs scores
    de similarité.
    """

    results = vector_store.similarity_search_with_score(
        question,
        k=k
    )


    print("Scores de similarité :")


    for doc, score in results:

        print(
            f"Score : {score:.4f} | "
            f"Source : {doc.metadata.get('source')}"
        )



# ==========================================================
# Fonction : Tester une question
# ==========================================================

def test_question(vector_store, retriever, question):

    print("\n" + "=" * 70)
    print(f"Question : {question}")
    print("=" * 70)


    # Recherche des chunks pertinents
    docs = retriever.invoke(question)


    print("\nChunks retrouvés :")
    display_documents(docs)


    # Affichage des scores
    display_similarity_scores(
        vector_store,
        question
    )



# ==========================================================
# Programme principal
# ==========================================================

def main():

    # Chargement de la base vectorielle
    vector_store = build_vectorstore()


    # Création du retriever
    retriever = create_retriever(
        vector_store,
        k=3
    )


    # Questions de test
    questions = [

        "Combien d'hôtels possède Apple Hospitality REIT au 31 décembre 2024 ?",


        "Quels sont les risques liés aux fournisseurs mentionnés par Tesla ?",


        "Pourquoi Wells Fargo a signalé un paiement en retard sur une carte de crédit inactive suite à un transfert de découvert non autorisé ?",


        "Quel problème de taxes foncières et d'assurance habitation est mentionné dans la réclamation Pulte Mortgage ?",


        "Quelle est la différence entre taux d'intérêt nominal et taux d'intérêt réel ?"

    ]


    # Test de toutes les questions
    for question in questions:

        test_question(
            vector_store,
            retriever,
            question
        )



# Point d'entrée du programme
if __name__ == "__main__":
    main()