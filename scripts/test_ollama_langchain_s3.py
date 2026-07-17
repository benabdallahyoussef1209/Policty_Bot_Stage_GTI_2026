from langchain_ollama import ChatOllama

llm = ChatOllama(model="phi3", temperature=0)
# llm = ChatOllama(model="ollama/phi3", temperature=0)  # alternative
question = input("Posez votre question : ")
reponse = llm.invoke(question)

print(f"Question : {question}")
print(f"Réponse : {reponse.content}")