from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

RAG_PROMPT = """
Réponds à la question en te basant uniquement sur le contexte ci-dessous.
Si tu ne trouves pas la réponse dans le contexte, dis "Je ne sais pas".

Contexte :
{context}

Question :
{question}

Réponse :
"""

prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
llm = ChatOllama(model="phi3", temperature=0)
chain = prompt | llm | StrOutputParser()

# --- Test 1 : la réponse EST dans le contexte ---
contexte_1 = "Apple Hospitality REIT possédait 221 hôtels au 31 décembre 2024."
question_1 = "Combien d'hôtels possède Apple Hospitality REIT ?"

reponse_1 = chain.invoke({"context": contexte_1, "question": question_1})
print("=== Test 1 : réponse présente dans le contexte ===")
print(f"Réponse : {reponse_1}\n")

# --- Test 2 : la réponse n'EST PAS dans le contexte ---
contexte_2 = "Apple Hospitality REIT possédait 221 hôtels au 31 décembre 2024."
question_2 = "Quel est le nom du PDG de l'entreprise ?"

reponse_2 = chain.invoke({"context": contexte_2, "question": question_2})
print("=== Test 2 : réponse absente du contexte ===")
print(f"Réponse : {reponse_2}")