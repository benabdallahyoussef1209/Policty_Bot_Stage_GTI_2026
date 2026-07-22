import sys
sys.path.insert(0, ".")

from src.ingest import build_vectorstore

vs = build_vectorstore()
print(f"Nombre de chunks dans la base : {vs._collection.count()}")