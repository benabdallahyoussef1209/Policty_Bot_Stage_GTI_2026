# Importe le module sys pour modifier le chemin de recherche des modules Python.
import sys

# Importe le module os pour manipuler les chemins de fichiers.
import os

# Ajoute automatiquement le dossier parent au chemin de recherche.
# Cela permet d'importer les fichiers présents dans le dossier "src"
# même si ce script est exécuté depuis le dossier "scripts".
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Importe la fonction permettant de créer ou charger la base vectorielle Chroma.
from src.ingest import build_vectorstore

# Charge la base vectorielle contenant les embeddings des documents.
vector_store = build_vectorstore()

# Crée un retriever.
# Le paramètre k=3 indique que les 3 chunks les plus pertinents seront retournés.
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Questions servant à tester le système RAG.
questions = [
    "Combien d'hôtels possède Apple Hospitality REIT au 31 décembre 2024 ?",
    "Quels sont les risques liés aux fournisseurs mentionnés par Tesla ?",
    "Pourquoi Wells Fargo a signalé un paiement en retard sur une carte de crédit inactive suite à un transfert de découvert non autorisé ?",
    "Quel problème de taxes foncières et d'assurance habitation est mentionné dans la réclamation Pulte Mortgage ?",
    "Quelle est la différence entre taux d'intérêt nominal et taux d'intérêt réel ?",
]
# Parcourt chaque question.
for question in questions:

    # Affiche la question en cours.
    print(f"\n=== {question} ===")

    # Recherche les 3 chunks les plus pertinents.
    docs = retriever.invoke(question)

    # Affiche chaque chunk retrouvé.
    for i, doc in enumerate(docs):

        # Affiche le numéro du chunk ainsi que son document d'origine.
        print(f"--- Chunk {i+1} (source: {doc.metadata.get('source')}) ---")

        # Affiche les 200 premiers caractères du chunk.
        print(doc.page_content[:200])

        # Ligne vide pour une meilleure lisibilité.
        print()

    # -------------------------------------------------------
    # Recherche avec calcul du score de similarité
    # -------------------------------------------------------

    # Retourne les 5 documents les plus proches ainsi que leur score.
    resultats = vector_store.similarity_search_with_score(question, k=3)

    print("Scores de similarité :")

    # Parcourt chaque résultat.
    for doc, score in resultats:

        # Affiche le score obtenu et le document source.
        print(
            f"Score : {score:.4f} | Source : {doc.metadata.get('source')}"
        )

    print("-" * 70)