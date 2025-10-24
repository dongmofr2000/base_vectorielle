import os
from langchain_community.document_loaders import DirectoryLoader
# L'importation corrigée
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration des chemins et des noms de fichiers
PERSIST_DIR = "faiss_index"
DOCS_DIR = "docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_documents():
    """
    Charge les documents du dossier DOCS_DIR. 
    Supporte les fichiers .txt, .pdf, .docx, et .csv.
    """
    print(f"Tentative de chargement des documents depuis le répertoire '{DOCS_DIR}'...")
    
    # Mapping des extensions vers les loaders correspondants
    loader_mapping = {
        ".txt": DirectoryLoader(DOCS_DIR, glob="**/*.txt", silent_errors=True),
        ".pdf": DirectoryLoader(DOCS_DIR, glob="**/*.pdf", silent_errors=True),
        ".docx": DirectoryLoader(DOCS_DIR, glob="**/*.docx", silent_errors=True),
        ".csv": DirectoryLoader(DOCS_DIR, glob="**/*.csv", silent_errors=True),
    }

    all_docs = []
    found_docs = False
    
    # Parcourt les loaders pour collecter tous les documents
    for ext, loader in loader_mapping.items():
        try:
            # Note: Si un loader est un DirectoryLoader, nous devons appeler .load()
            current_docs = loader.load()
            if current_docs:
                all_docs.extend(current_docs)
                found_docs = True
                print(f"  -> {len(current_docs)} document(s) {ext} chargé(s).")
        except Exception as e:
            # Gestion des erreurs d'importation de bibliothèques externes non installées (ex: pypdf)
            # Nous affichons ici une erreur pour le débogage si un loader échoue.
            print(f"Erreur lors du chargement des fichiers {ext} (cela peut être dû à un loader manquant): {e}")

    if not found_docs:
        print(f"AVERTISSEMENT: Aucun document trouvé dans le répertoire '{DOCS_DIR}'.")
        print("Veuillez créer un dossier 'docs' et y placer au moins un document source (ex: .txt, .pdf).")
        return []
        
    return all_docs

def split_documents(documents):
    """
    Découpe les documents en morceaux (chunks) pour le RAG.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Documents découpés en {len(chunks)} chunks.")
    return chunks

def create_and_save_vector_store(chunks):
    """
    Crée les embeddings, génère la base de données vectorielle FAISS et la sauvegarde.
    """
    print(f"Chargement du modèle d'embeddings: {MODEL_NAME}")
    # Pour s'assurer que le modèle est téléchargé si non présent
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    print("Création de la base de données vectorielle FAISS...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    print(f"Sauvegarde de l'index dans le répertoire '{PERSIST_DIR}'...")
    vector_store.save_local(PERSIST_DIR)
    print("Base de données vectorielle créée et sauvegardée avec succès.")


if __name__ == "__main__":
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"Dossier '{DOCS_DIR}' créé. Veuillez y ajouter vos fichiers PDF/TXT.")
    else:
        print("--- Démarrage du processus d'ingestion ---")
        
        # 1. Chargement des documents
        documents = load_documents()
        
        if documents:
            # 2. Découpage en chunks
            chunks = split_documents(documents)
            
            # 3. Création et sauvegarde de la base de données vectorielle
            create_and_save_vector_store(chunks)

        print("--- Processus d'ingestion terminé ---")
