import os
import getpass
from langchain_community.document_loaders import TextLoader # Nouvelle structure d'importation
from langchain_text_splitters import RecursiveCharacterTextSplitter # CORRECTION: Nouveau chemin pour le TextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings # Nouvelle structure d'importation
from langchain_community.vectorstores import Chroma

# --- Configuration et Initialisation ---

# Assurez-vous d'avoir un fichier de données dans le même répertoire.
# Remplacer 'data.txt' par le nom de votre fichier de données (e.g., 'documents.txt').
DATA_FILE_PATH = "data.txt" 
VECTOR_DB_DIR = "./chroma_db"

# 1. Charger le Document
try:
    print(f"Chargement des données depuis {DATA_FILE_PATH}...")
    # Assurez-vous que ce fichier existe ou remplacez-le par le bon chemin.
    loader = TextLoader(DATA_FILE_PATH, encoding="utf8")
    documents = loader.load()
    print(f"Chargé {len(documents)} document(s).")
except FileNotFoundError:
    print(f"ERREUR: Le fichier {DATA_FILE_PATH} est introuvable. Veuillez créer ce fichier.")
    exit()
except Exception as e:
    print(f"Une erreur est survenue lors du chargement: {e}")
    exit()

# 2. Diviser le Texte (Chunking)
# Le chunking est essentiel pour le RAG: il divise le texte long en petits morceaux
print("Division des documents en morceaux (chunks)...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
print(f"Création de {len(texts)} morceaux de texte pour l'indexation.")

# 3. Choisir le Modèle d'Embedding
# Utilisation d'un modèle d'embedding de HuggingFace (par exemple, 'all-MiniLM-L6-v2')
# Ce modèle convertit le texte en vecteurs.
print("Initialisation du modèle d'embedding HuggingFace...")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# 4. Créer et Persister le Vector Store (Chroma)
# Chroma est la base de données qui stocke les vecteurs et les associe au texte original.
print(f"Création de l'index vectoriel Chroma dans le répertoire: {VECTOR_DB_DIR}...")
try:
    db = Chroma.from_documents(
        texts, 
        embeddings, 
        persist_directory=VECTOR_DB_DIR
    )
    db.persist() # Sauvegarde l'index sur le disque
    print("Indexation terminée avec succès ! La base de données est prête pour la recherche.")
    print("\n--- PROCHAINE ÉTAPE ---")
    print("Vous pouvez maintenant créer le script de recherche (rag_query.py) pour effectuer vos tests d'efficacité.")

except Exception as e:
    print(f"ÉCHEC DE L'INDEXATION: {e}")
    print("Vérifiez les dépendances (pip install torch) ou les droits d'accès au répertoire.")

# Note : Si vous n'avez pas de fichier 'data.txt', ce script affichera une erreur.
# Créez un simple fichier 'data.txt' avec du texte pour tester.
