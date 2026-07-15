import sqlite3

conn = sqlite3.connect("chroma_db/chroma.sqlite3")
cursor = conn.cursor()

# 1. Liste toutes les tables présentes dans la base
print("=== Tables dans chroma.sqlite3 ===")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
for t in tables:
    print(t[0])

print("\n=== Contenu de la table 'collections' ===")
cursor.execute("SELECT * FROM collections;")
for row in cursor.fetchall():
    print(row)

print("\n=== Exemple de 3 lignes de la table 'embeddings' (métadonnées) ===")
cursor.execute("SELECT * FROM embeddings LIMIT 3;")
for row in cursor.fetchall():
    print(row)

print("\n=== Contenu de 'embedding_metadata' (le vrai texte) ===")
cursor.execute("SELECT * FROM embedding_metadata LIMIT 5;")
for row in cursor.fetchall():
    print(row)
