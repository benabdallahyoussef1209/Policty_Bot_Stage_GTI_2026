from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:1b")
response = llm.invoke("Dis bonjour en une phrase")
print(response.content)