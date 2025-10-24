# ====================================================================
#              PROJET RAG (RETRIEVAL-AUGMENTED GENERATION)
# ====================================================================

# 🌟 Présentation du Projet
# -------------------------
# Ce projet implémente un système de Génération Augmentée par Récupération (RAG) simplifié.
# L'objectif est de fournir à un Grand Modèle de Langage (LLM) un contexte pertinent 
# (récupéré d'une base de connaissances locale) avant de générer une réponse.

# Le système est divisé en deux étapes :
# 1. Indexation : Création d'un index vectoriel FAISS à partir du document source (evenements_reels.txt).
# 2. Récupération : Recherche des morceaux de texte les plus similaires à une requête utilisateur.

# 🔧 Instructions pour la Reproduction
# ------------------------------------

# Étape 1 : Installation des Dépendances
# Vous devez avoir Python (3.8+) et installer les bibliothèques suivantes :
# pip install langchain-community langchain-core langchain-huggingface langchain-text-splitters faiss-cpu

# Étape 2 : Indexation (Création de la Base de Connaissances)
# Lancez le script vector_indexer.py. Il va lire evenements_reels.txt, le découper,
# créer les embeddings, et sauvegarder l'index vectoriel FAISS dans un dossier ./faiss_index.
# COMMANDE : python vector_indexer.py

# Étape 3 : Récupération (Interrogation de l'Index)
# Une fois l'index créé, lancez le script .\venv\Scripts\python.exe chat_rag.py --doc evenements_reels.pdf. Il chargera l'index et recherchera
# les 3 morceaux de texte les plus pertinents pour répondre à la question.
# COMMANDE : .\venv\Scripts\python.exe chat_rag.py --doc evenements_reels.pdf

# ====================================================================
#              FIN DU CONTENU README
# ====================================================================
