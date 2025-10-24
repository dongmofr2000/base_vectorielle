import os
# Nouvelle importation corrigée pour CharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS # Utilisation de FAISS

# --- Configuration ---
SOURCE_FILE = "evenements_reels.txt"
# Chemin où FAISS sauvegardera l'index et les métadonnées
INDEX_PATH = "./faiss_index" 
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- 1. Chargement des données ---

print(f"Chargement des données depuis {SOURCE_FILE}...")
loader = TextLoader(SOURCE_FILE, encoding='utf-8')
documents = loader.load()
print(f"Chargé {len(documents)} document(s).")

# --- 2. Division en morceaux (Chunking) ---

print("Division des documents en morceaux (chunks)...")
# Utiliser un séparateur de texte simple pour l'exemple
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
print(f"Création de {len(texts)} morceaux de texte pour l'indexation.")

# --- 3. Initialisation du modèle d'embedding ---

print("Initialisation du modèle d'embedding HuggingFace...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# --- 4. Création de l'index vectoriel FAISS ---

print(f"Création de l'index vectoriel FAISS dans le répertoire: {INDEX_PATH}...")

# Créer l'index FAISS à partir des documents et des embeddings
db = FAISS.from_documents(texts, embeddings)

# S'assurer que le répertoire de sauvegarde existe
if not os.path.exists(INDEX_PATH):
    os.makedirs(INDEX_PATH)
    
# Sauvegarder l'index localement
db.save_local(INDEX_PATH)

print("\nIndexation FAISS terminée avec succès !")
print(f"L'index FAISS est prêt et sauvegardé dans le dossier '{INDEX_PATH}'.")

# --- Note Finale ---
print("\nL'index a été construit uniquement à partir du fichier '{SOURCE_FILE}'.")
