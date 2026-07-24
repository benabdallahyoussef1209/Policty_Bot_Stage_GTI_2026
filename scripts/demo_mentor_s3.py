import sys
sys.path.insert(0, ".")

from src.ingest import build_vectorstore
from src.chain import ask

print("=" * 70)
print("PolicyBot — Démonstration Semaine 3")
print("=" * 70)

vector_store = build_vectorstore()

questions_demo = [
    ("Réponse précise avec source",
     "Quelle est la différence entre taux d'intérêt nominal et taux d'intérêt réel ?"),

    ("Absence d'hallucination (question hors-sujet)",
     "Quelle est la capitale de la France ?"),

    ("Extraction sur document dense (85 000+ caractères)",
     "Quels sont les risques liés à la chaîne d'approvisionnement de Tesla ?"),

    ("Cas limite étudié en profondeur (dilution de similarité)",
     "Combien d'hôtels possède Apple Hospitality REIT au 31 décembre 2024 ?"),
]

for i, (titre, question) in enumerate(questions_demo, 1):
    print(f"\n--- Démo {i} : {titre} ---")
    print(f"Question : {question}")
    reponse, sources = ask(question, vector_store)
    print(f"Réponse  : {reponse}")
    print(f"Source(s): {', '.join(sources)}")
    print("-" * 70)

print("\nDémonstration terminée.")