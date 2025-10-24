import argparse
import os
import sys
from dotenv import load_dotenv

# --- Imports corrigés pour LangChain 1.x ---
# Les anciennes fonctions ont été déplacées vers langchain_core ou d'autres packages spécifiques.
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------

# Configuration de base
# Assurez-vous d'avoir MISTRAL_API_KEY dans votre fichier .env
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    print("Erreur: La variable d'environnement MISTRAL_API_KEY n'est pas définie.")
    sys.exit(1)

# Configuration du modèle
LLM = ChatMistralAI(model="mistral-large-latest", api_key=MISTRAL_API_KEY)
EMBEDDING_MODEL = MistralAIEmbeddings(model="mistral-embed", api_key=MISTRAL_API_KEY)

# ---------------------------------------------
# 1. Pipeline de Traitement du Document
# ---------------------------------------------

def process_document_and_get_retriever(doc_path: str):
    """
    Charge un document PDF, le divise en chunks, l'intègre et crée un Retriever.
    """
    print(f"Chargement du document: {doc_path}")
    
    # 1. Charger le document
    try:
        loader = PyPDFLoader(doc_path)
        documents = loader.load()
    except Exception as e:
        print(f"Erreur lors du chargement du document: {e}")
        return None

    # 2. Diviser le texte
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    print(f"Document divisé en {len(splits)} morceaux (chunks).")

    # 3. Créer le VectorStore et le Retriever (stockage temporaire en mémoire)
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=EMBEDDING_MODEL
    )
    
    print("VectorStore (Chroma) créé avec succès.")
    # On retourne le retriever pour l'utiliser dans la chaîne RAG
    return vectorstore.as_retriever()


# ---------------------------------------------
# 2. Pipeline de Chat (LCEL)
# ---------------------------------------------

def format_docs(docs):
    """Simple fonction utilitaire pour formater les documents en une seule chaîne."""
    # Le contenu est séparé par une double ligne pour l'IA
    return "\n\n".join(doc.page_content for doc in docs)

# 1. Définir le Template de Prompt pour le RAG
template = """
Vous êtes un assistant IA utile et précis. 
Utilisez les morceaux de contexte récupérés ci-dessous pour répondre à la question. 
Si vous ne trouvez pas la réponse dans le contexte, dites que vous n'avez pas assez d'informations.

Contexte:
{context}

Question: {question}
Réponse:
"""
prompt = ChatPromptTemplate.from_template(template)


def create_rag_chain(retriever):
    """Crée la chaîne RAG complète en utilisant LCEL (LangChain Expression Language)."""
    
    # La chaîne entière est construite ici
    rag_chain = (
        # 1. Récupère le contexte à partir de la question de l'utilisateur
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        # 2. Assemble le prompt complet
        | prompt
        # 3. Appelle le modèle LLM
        | LLM
        # 4. Extrait la sortie sous forme de chaîne simple
        | StrOutputParser()
    )
    
    print("Chaîne RAG créée. Prêt à chatter.")
    return rag_chain

# ---------------------------------------------
# 3. Fonction Principale
# ---------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Un chatbot RAG (Retrieval Augmented Generation) basé sur un document PDF et Mistral AI.")
    parser.add_argument("--doc", required=True, help="Chemin d'accès au document PDF à utiliser (ex: docs/mon_fichier.pdf)")
    args = parser.parse_args()

    # Étape 1: Traitement et Indexation
    retriever = process_document_and_get_retriever(args.doc)

    if retriever is None:
        print("Échec de la préparation du RAG. Arrêt.")
        return

    # Étape 2: Création de la Chaîne
    rag_chain = create_rag_chain(retriever)
    
    # Étape 3: Boucle de Chat
    print("\n--- Démarrage de la session de Chat ---")
    print("Posez vos questions basées sur le document. Tapez 'quitter' pour terminer.")

    while True:
        try:
            user_input = input("Vous > ")
            if user_input.lower() in ['quitter', 'exit']:
                print("Fin de la session de chat. Au revoir!")
                break
            
            if not user_input.strip():
                continue

            # Appel de la chaîne RAG
            print("\nIA (Réflexion...)")
            response = rag_chain.invoke(user_input)
            
            # Affichage de la réponse
            print(f"\nIA > {response}\n")

        except KeyboardInterrupt:
            print("\nFin de la session de chat. Au revoir!")
            break
        except Exception as e:
            # Note: Si vous rencontrez un 'rate limit error' ou une autre erreur API, elle sera affichée ici.
            print(f"\nUne erreur inattendue est survenue: {e}")
            break

if __name__ == "__main__":
    main()
