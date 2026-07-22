import sys
sys.path.insert(0, ".")

from src.ingest import build_vectorstore
from src.chain import ask

vector_store = build_vectorstore()

question_1 = "Combien d hotels possède Apple Hospitality REIT au 31 décembre 2024 ?"
question_7 = "Quelle est la capitale de la France ?"
question_4 = "Quel problème de taxes foncières est mentionné dans la réclamation Pulte Mortgage ?"
for question in [question_1, question_7, question_4]:
    reponse, sources = ask(question, vector_store)
    print(f"=== {question} ===")
    print(f"Réponse : {reponse}")
    print(f"Sources : {sources}")
    print("-" * 70)
    # Ajoute ceci à ton script de test
docs = vector_store.similarity_search_with_score(
    "Combien d'hôtels possède Apple Hospitality REIT au 31 décembre 2024 ?", k=20
)