import sys
sys.path.insert(0, ".")

from src.ingest import build_vectorstore
from src.chain import ask

vector_store = build_vectorstore()
question = "Combien d'hôtels possède Apple Hospitality REIT ?"
reponse, sources = ask(question, vector_store)

print(f"Question : {question}")
print(f"Réponse : {reponse}")
print(f"Sources : {sources}")