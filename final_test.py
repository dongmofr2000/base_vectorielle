import os
import sys

# Vérification des importations modernes (V1.x)
try:
    import langchain
    
    # *** CORRECTION MAJEURE : Importer les modèles directement depuis le paquet dédié ***
    from langchain_mistralai import MistralAIEmbeddings
    from langchain_mistralai import ChatMistralAI as MistralAI
    
    from langchain_core.prompts import ChatPromptTemplate
    
    print("--- Test de l'environnement LangChain (V1.x) ---")
    print(f"✅ Succès : Le paquet 'langchain' est importé (v{langchain.__version__})")
    print("✅ Succès : 'langchain_core.prompts' est importé.")
    print("✅ Succès : L'Embeddings MistralAI est importé (chemin direct : langchain_mistralai).")
    print("✅ Succès : Le modèle LLM MistralAI est importé (chemin direct : langchain_mistralai).")

    # Test de la configuration pour un composant RAG commun
    try:
        from langchain_community.vectorstores import FAISS
        print("✅ Succès : 'langchain_community.vectorstores' est importé (FAISS).")
    except ImportError as e:
        print(f"⚠️ Avertissement : FAISS n'a pas pu être importé. Erreur: {e}")

    print("\n------------------------------------------------------------------")
    print("🚀 FÉLICITATIONS ! Votre environnement est prêt pour le développement RAG.")
    print("------------------------------------------------------------------")

except ImportError as e:
    print(f"❌ ÉCHEC FATAL : Un module critique est manquant. Erreur: {e}")
    sys.exit(1)
