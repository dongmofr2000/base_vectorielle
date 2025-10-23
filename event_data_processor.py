import csv 
import re 
import io 

# --- CONVERSION EN DICTIONNAIRE PYTHON NATIF POUR BYPASSER L'ERREUR DE PARSING JSON ---
# Cette structure est un dictionnaire Python standard, éliminant toute erreur de syntaxe de chaîne.
raw_data = {
"total_count": 904209,
"results": [
    {
        "uid": "12413663",
        "title_fr": "Centre social St Rambert d'Albon",
        "description_fr": "organisé par Centre social St Rambert d'Albon",
        "longdescription_fr": "<p>Atelier de danses \"Folk\"</p>...",
        "conditions_fr": "Gratuit",
        "daterange_fr": "Mardi 25 mai 2021, 19h00",
        "location_name": "19 Avenue de Lyon, Saint-Rambert-d'Albon (26)",
        "location_city": None
    },
    {
        "uid": "75551745",
        "title_fr": "Coupe de Normandie VTT - 25 septembre 2021 - Feuguerolles-Bully",
        "description_fr": "Coupe de Normandie et Trophée Régional des Jeunes Vététistes",
        "conditions_fr": "pass sanitaire obligatoire",
        "daterange_fr": "Samedi 25 septembre 2021, 10h00",
        "location_name": "Feuguerolles Bully",
        "location_city": "Feuguerolles-Bully"
    },
    {
        "uid": "87153658",
        "title_fr": "Un, deux, trad invite les Tralala Lovers",
        "description_fr": "avec Tralala Lovers et Viendez-Voir",
        "conditions_fr": "10€",
        "daterange_fr": "Samedi 6 novembre 2021, 21h00",
        "location_name": "Avenue Charles Duchêne, Mirecourt (88)",
        "location_city": None
    },
    {
        "uid": "35021962",
        "title_fr": "Totalement 90",
        "description_fr": "WORLDS APART, LARUSSO, INDRA, BORIS, YANNICK En maîtres de cérémonie, Charly et Lulu sauront à coup sûr mettre le feu lors de cette soirée « Totalement 90 ».",
        "conditions_fr": "Tarifs: de 35€ à 45€",
        "daterange_fr": "Jeudi 18 novembre 2021, 20h00",
        "location_name": "Théâtre de Longjumeau",
        "location_city": "Longjumeau"
    },
    {
        "uid": "71691788",
        "title_fr": "Atelier « Signer avec bébé » avec Laure Giry",
        "description_fr": "Séance N°1 sur 2, pour découvrir la communication gestuelle adaptée aux tout-petits !",
        "conditions_fr": "Sur réservation auprès d’A3N",
        "daterange_fr": "Samedi 3 novembre 2018, 14h00",
        "location_name": "Agora",
        "location_city": "Issy-les-Moulineaux"
    },
    {
        "uid": "84172770",
        "title_fr": "Spectacle : Cantique des Cantiques, Songes de Leonard Cohen",
        "description_fr": "Spectacle conçu pour une pièce musicale mêlant influences ethniques, rock alternatif et atmosphères néo-classiques, méditation de Leonard Cohen",
        "conditions_fr": "Entrée libre, sur réservation en ligne",
        "daterange_fr": "Samedi 16 mars 2019, 18h00",
        "location_name": "Espace Andrée Chedid",
        "location_city": "Issy-les-Moulineaux"
    },
    {
        "uid": "14522599",
        "title_fr": "Les artistes de Nahariya exposent",
        "description_fr": "Exposition",
        "conditions_fr": "Entrée libre",
        "daterange_fr": "19 juin - 1 juillet 2018",
        "location_name": "Médiathèque centre-ville",
        "location_city": "Issy-les-Moulineaux"
    }
]
}

def process_event_data(data_dict, output_filename='extracted_data.csv'):
    """
    Traite les données du dictionnaire Python d'événements et les exporte
    dans un fichier CSV structuré pour la vectorisation.
    """
    print("--- DÉMARRAGE DU TRAITEMENT ET EXPORT CSV ---")
    
    try:
        # Accès direct aux résultats puisque 'data_dict' est déjà un dictionnaire
        events = data_dict.get('results', [])
        
        # 1. Définir les noms des colonnes pour le CSV
        fieldnames = ['uid', 'title', 'date_range', 'location', 'conditions', 'full_text']
        
        # 2. Ouvrir le fichier CSV en mode écriture
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            # Utilisation du délimiteur ';' pour éviter les conflits avec les virgules dans le texte
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';') 
            writer.writeheader() 
            
            # 3. Traiter et écrire chaque événement
            for event in events:
                # Créer un champ 'full_text' combiné pour la vectorisation
                # Remplacer None (valeur Python) par une chaîne vide pour la concaténation
                location_city_str = event.get('location_city') or 'Ville non spécifiée'
                
                full_text_description = (
                    f"Titre de l'événement: {event.get('title_fr', 'Titre non disponible')}. "
                    f"Description courte: {event.get('description_fr', '')}. "
                    f"Date: {event.get('daterange_fr', '')}. "
                    f"Lieu: {event.get('location_name', '')} à {location_city_str}. "
                    f"Conditions d'accès/Prix: {event.get('conditions_fr', '')}. "
                    f"Description complète: {event.get('longdescription_fr', '')}"
                )
                
                # Nettoyage : retirer les balises HTML et les espaces multiples
                full_text_cleaned = re.sub(r'<[^>]+>', '', full_text_description) # Retire les balises HTML
                full_text_cleaned = re.sub(r'\s+', ' ', full_text_cleaned).strip() # Retire les espaces multiples
                
                # Nettoyage de la chaîne de localisation pour l'export CSV
                location_str = f"{event.get('location_name', '')}, {location_city_str}"
                location_str = location_str.replace('None', '').replace('Ville non spécifiée', '').strip(', ')

                row_data = {
                    'uid': event.get('uid', ''),
                    'title': event.get('title_fr', 'Titre non disponible'),
                    'date_range': event.get('daterange_fr', 'Date non spécifiée'),
                    'location': location_str,
                    'conditions': event.get('conditions_fr', ''),
                    'full_text': full_text_cleaned
                }
                
                writer.writerow(row_data) 

        print(f"--- TRAITEMENT TERMINÉ ---")
        print(f"Le fichier '{output_filename}' contenant les données des {len(events)} événements a été créé avec succès. ")
        
    except Exception as e:
        print(f"Une erreur inattendue s'est produite lors du traitement : {e}")

# Exécuter la fonction de traitement avec le dictionnaire natif
process_event_data(raw_data)
