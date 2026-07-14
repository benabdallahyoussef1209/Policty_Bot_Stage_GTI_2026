import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collections = client.list_collections()
for c in collections:
    print(c.name)