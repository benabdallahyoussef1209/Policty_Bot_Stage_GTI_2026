import sys
sys.path.insert(0, ".")

from src.ingest import build_vectorstore
from src.chain import ask

vector_store = build_vectorstore()

questions = [
    "Combien d'hôtels possède Apple Hospitality REIT au 31 décembre 2024 ?",
    "Quels sont les risques liés à la chaîne d'approvisionnement de Tesla ?",
    "Pourquoi Wells Fargo a signalé un paiement en retard sur une carte de crédit inactive ?",
    "Quel problème de taxes foncières est mentionné dans la réclamation Pulte Mortgage ?",
    "Quelle est la différence entre taux d'intérêt nominal et taux d'intérêt réel ?",
    "Quel montant MOHELA devait-il rembourser suite à l'erreur de solde du prêt étudiant ?",
    "Quelle est la capitale de la France ?",
    "Quel est le meilleur restaurant de Tunis ?",
    "Comment fait-on une omelette ?",
    "Quels sont les risques ?",
]

for i, question in enumerate(questions, 1):
    reponse, sources = ask(question, vector_store)
    print(f"=== Question {i} : {question} ===")
    print(f"Réponse : {reponse}")
    print(f"Sources : {sources}")
    print("-" * 70)