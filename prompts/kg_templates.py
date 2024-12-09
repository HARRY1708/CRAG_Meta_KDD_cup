FINANCE_TEMPLATE

MOVIE_TEMPLATE

MUSIC_TEMPLATE

OPEN_TEMPLATE

SPORTS_TEMPLATE


########################################################################################
# Classification prompt:
########################################################################################

CLASSIFICATION_SYSTEM_PROMPT = """You are a highly accurate classifier that categorizes user questions into two categories: "domain" or "other".

## Instructions (Read it Very Carefully):
Your task is to categorize the input question given by the user in one of the following categories:
    - Domain 
    - Other
    For deciding the categories for the given inupt question following are categorization crietria for "domain":
    
        1. If the given input question from user belongs to one of the following categories (provided in the format - [category: deciding criteria for the category]) then classify the given input question as "domain":
        - Sports: If the main theme of the given input question form the user is around sports for example if the given input question is around a sports team or talks about a sports player then it belongs to sports and hence "domain". 
        - Finance: If the main theme of the given input question form the user is around sports for example if the given input question is around a sports team or talks about a sports player then it belongs to finance and hence "domain".
        - Music: If the main theme of the given input question form the user is around sports for example if the given input question is around a sports team or talks about a sports player then it belongs to music and hence "domain".
        - Movie: If the main theme of the given input question form the user is around sports for example if the given input question is around a sports team or talks about a sports player then it belongs to movie and hence "domain".
        - Open: If the main theme of the given input question form the user is around sports for example if the given input question is around a sports team or talks about a sports player then it belongs to open and hence "domain".
        
    For deciding the categories for the given inupt question following are categorization crietria for "other":

A question is classified as "domain" if its main theme belongs to one of the following categories:
- Sports
- Finance
- Music
- Movie
- Open

If the main theme of the question does not belong to any of the above categories, classify it as "other".
When classifying, consider the primary context or subject of the question. Make sure the output response if only with the category: "domain" or "other". If the given input question belogns to the "domain" category in that case make sure to return the category to which the input question belongs to as well. For example: 

IN-CONTEXT_EXAMPLE
"""
CLASSIFICATION_PROMPT = f"""{CLASSIFICATION_SYSTEM_PROMPT}

User query: {query}
Given the above user query, classify it in one of the following classes: "domain" or "other". 
In context examples: 
- IN-CONTEXT EXAMPLE #1:

- IN-CONTEXT EXAMPLE #2:
"""

########################################################################################
# NER prompts:
########################################################################################
NER_SYSTEM_PROMPT = """ 

""""

NER_SPORTS_PROMPT = f""" Input question: {input_question} 
Identify and list all the named entities present in the above-mentioned input question about sports and make SURE to NOT answer the question and list all the entities present in the question like nfl team, nba team, soccer team, nfl player, nba player, soccer player, etc. If there are no entities present in the question then in that case jsut output "No entities". The output format should be in the following format:
{
    "team_name": "name of the nfl, nba or soccer team in the input question",
    "player_name": "name of the nba, nfl or soccer player in the input question",
    "date": "date being referenced in the question, make sure that the format fo the ouput date is MM-DD-YYYY or if it a range YYYY to YYYY should the required format"
}

In the above format if any of the above mentioned categories are not present in the inpupt question then jsut ouput "None" for that category.
"""

NER_MOVIE_PROMPT = f""" Input question: {input_question}
Identify and list all the named entities present in the above-mentioned input question about movies and make SURE to NOT answer the question and list all the entities present in the question like movie name, person in the movie, actors in the movie, cast in the movie, etc. If there are no entities present in the question then in that case jsut output "No entities". The output format should be in the following format:
{
    "person_name": "name of a person or the character being referenced in the input question", 
    "movie_name": "name of a movie in the input question",
    "year": "if year is being referenced in the question then the requrired format is YYYY and date being referenced in the question, make sure that the format fo the ouput date is MM-DD-YYYY or if it a range YYYY to YYYY should the required format"
}
In the above format if any of the above mentioned categories are not present in the inpupt question then jsut ouput "None" for that category.
"""

NER_FINANCE_PROMPT = f""" Input question: {input_question}
Identify and list all the named entities present in the above-mentioned input question about finance and make SURE to NOT answer the question and list all the entities present in the question like company name, stock name, corporations, ticker symbols etc. If there are no entities present in the question then in that case jsut output "No entities". The output format should be in the following format:
{
    "compnay_name": "name of the company being referenced in the input question",
    "ticker_name": "name of the ticker (unique combination of letters and numbers used to identify a publicly traded security, like a stock, on a stock exchange) being referenced in the input question."
}

In the above format if any of the above mentioned categories are not present in the inpupt question then jsut ouput "None" for that category.
"""

NER_OPEN_PROMPT = f""" Input question: {input_question} 
Identify and list all the named entities present in the above-mentioned input question and make SURE to NOT answer the question :
{
    "person_name": "name of a person being referenced in the input question",
    "location_name": "name of a location in the input question",
    "organization_name": "name of an organization in the input question",
    "product_name": "name of a product in the input question",
    "event_name": "name of an event in the input question"
}
In the above format if any of the above mentioned categories are not present in the inpupt question then jsut ouput "None" for that category.
"""

NER_MUSIC_PROMPT = f""" Input question: {input_question}
Identify and list all the named entities present in the above-mentioned input question about music and make SURE to NOT answer the question and list all the entities present in the question like movie name, person in the song name, singer name, song writer name, band name, artisit name, etc. If there are no entities present in the question then in that case jsut output "No entities". The output format should be in the following format:
{
    "artist_name": "name of a person in the input question",
    "song_name": "name of a song in the input question",
    "date": "if year is being referenced in the question then the requrired format is YYYY and date being referenced in the question, make sure that the format fo the ouput date is MM-DD-YYYY or if it a range YYYY to YYYY should the required format",
    "band_nmae": "name of a band in the input question"
}

In the above format if any of the above mentioned categories are not present in the inpupt question then jsut ouput "None" for that category.
"""


########################################################################################
# generation prompts:
########################################################################################

FINANCE_TEMPLATE = """

"""

MOVIE_TEMPLATE = """

"""

MUSIC_TEMPLATE = """

"""

OPEN_TEMPLATE = """

"""

SPORTS_TEMPLATE = """

"""





def kg_context_retrieval(domain, entities):
    kg_results = []
    if domain.strip().lower == "movie":
        if entities['person_name'] is not '':
            kg_result = movie_get_person_info(person_name: str)
            kg_results.apppend(kg_result)
        if entities['movie_name'] is not '':
            kg_result = movie_get_movie_info(movie_name: str) 
            kg_results.apppend(kg_result)
        if entities['year'] is not '':
            kg_result = movie_get_year_info(year: str) 
            kg_results.apppend(kg_result)
        if entities['movie_id'] is not '':
            kg_result = movie_get_movie_info_by_id(movie_id: int)
            kg_results.apppend(kg_result)
        if entities['person_id'] is not '':
            kg_result = movie_get_person_info_by_id(person_id: int)
            kg_results.apppend(kg_result)
        
    elif domain.strip().lower == "sports":
        if entities['team_name'] is not '' and entities['date'] is not '':
            kg_result = sports_soccer_get_games_on_date(team_name: str, date: str) 
            kg_results.append(kg_result)
        if entities['team_name'] is not '' and entities['date'] is not '':
            kg_result = sports_nba_get_games_on_date(team_name: str, date: str)
            kg_results.append(kg_result)
        if entities['games_id'] is not '':
            kg_result = sports_nba_get_play_by_play_data_by_game_ids(game_ids: List[str])
            kg_results.append(kg_result)
        
        
    elif domain.strip().lower == "open":
        if entities['entity_name'] is not '':
            kg_result = open_search_entity_by_name(query: str)
            kg_results.append(kg_result)
        if entities['entity'] is not '' or entities['entity_name'] is not '':
            kg_result = open_get_entity(entity: str)
            kg_results.append(kg_result)
        
    elif domain.strip().lower == "music":
        if entities['artist_name'] is not '':
            kg_result = music_search_artist_entity_by_name(artist_name: str) 
            kg_results.append(kg_result)
        if entities['song_name'] is not '':
            kg_result = music_search_song_entity_by_name(song_name: str) 
            kg_results.append(kg_result)
        if entities['rank'] is not None and entities['date'] is not '':
            kg_result = music_get_billboard_rank_date(rank: int, date: str = None) 
            kg_results.append(kg_result)
        if entities['date'] is not '' and entities['attribute'] is not '' and entities['song_name'] is not '':
            kg_result = music_get_billboard_attributes(date: str, attribute: str, song_name: str) 
            kg_results.append(kg_result)
        if entities['date'] is not '':
            kg_result = music_grammy_get_best_artist_by_year(year: int) 
            kg_results.append(kg_result)
        if entities['artist_name'] is not '':
            kg_result = music_grammy_get_award_count_by_artist(artist_name: str)
            kg_results.append(kg_result)
        if entities['song_name'] is not '':
            kg_result = music_grammy_get_award_count_by_song(song_name: str) 
            kg_results.append(kg_result)
        if entities['date'] is not '':
            kg_result = music_grammy_get_best_song_by_year(year: int) 
            kg_results.append(kg_result)
        if entities['artist_name'] is not '':
            kg_result = music_grammy_get_award_date_by_artist(artist_name: str)
            kg_results.append(kg_result)
        if entities['date'] is not '':
            kg_result = music_grammy_get_best_album_by_year(year: int) 
            kg_results.append(kg_result)
        if entities['artist_name'] is not '':
            kg_result = music_get_artist_birth_place(artist_name: str) 
            kg_results.append(kg_result)
        if entities['artist_name'] is not '':
            kg_result = music_get_artist_birth_date(artist_name: str) 
            kg_results.append(kg_result)
        if entities['band_name'] is not '':
            kg_result = music_get_members(band_name: str) 
            kg_results.append(kg_result)
        if entities['artist_name'] is not '':
            kg_result = music_get_lifespan(artist_name: str)
            kg_results.append(kg_result)
        if entities['song_name'] is not '':
            kg_result = music_get_song_author(song_name: str)
            kg_results.append(kg_result)
        if entities['song_name'] is not '':
            kg_result = music_get_song_release_country(song_name: str)
            kg_results.append(kg_result)
        if entities['song_name'] is not '':
            kg_result = music_get_song_release_date(song_name: str) 
            kg_results.append(kg_result)
        if entities['artist_name'] is not '':
            kg_result = music_get_artist_all_works(artist_name: str)
            kg_results.append(kg_result)
        kg_result = music_grammy_get_all_awarded_artists() 
        kg_results.append(kg_result)
        
    elif domain.strip().lower == "finance":
        if entities['company_name'] is not '':
            kg_result = finance_get_company_name(query: str)
            kg_results.append(kg_result)
        if entities['ticker_name'] is not '':
            kg_result = finance_get_ticker_by_name(query: str)
            kg_results.append(kg_result)
        if entities['ticker_name'] is not '':
            kg_result = finance_get_price_history(ticker_name: str) 
            kg_results.append(kg_result)
        if entities['ticker_name'] is not '':
            kg_result = finance_get_detailed_price_history(ticker_name: str) 
            kg_results.append(kg_result)
        if entities['ticker_name'] is not '':
            kg_result = finance_get_dividends_history(ticker_name: str) 
            kg_results.append(kg_result)
        if entities['ticker_name'] is not '':
            kg_result = finance_get_market_capitalization(ticker_name: str)
            kg_results.append(kg_result)
        if entities['ticker_name'] is not '':
            kg_result = finance_get_eps(ticker_name: str)
            kg_results.append(kg_result)
        if entities['ticker_name'] is not '':
            kg_result = finance_get_pe_ratio(ticker_name: str)
            kg_results.append(kg_result)
        if entities['ticker_name'] is not '':
            kg_result = finance_get_info(ticker_name: str)
            kg_results.append(kg_result)
        
    else:
        
        
    
    