def ner_sports_template(input_question):
    system_message = """Identify and list all the named entities present in the above-mentioned input question about sports and make SURE to NOT answer the question and list all the entities present in the question like nfl team, nba team, soccer team, nfl player, nba player, soccer player, etc. If there are no entities present in the question then in that case jsut output "No entities". The output format should be in the following format:
    Output: {
        "team_name": "name of the nfl, nba or soccer team in the input question",
        "player_name": "name of the nba, nfl or soccer player in the input question",
        "date": "date being referenced in the question, make sure that the format fo the ouput date is MM-DD-YYYY or if it a range YYYY to YYYY should the required format"
    }

    Make sure to include "Output:" before the dictionary. In the above format if any of the above mentioned categories are not present in the inpupt question then just ouput '' (Empty String) for that category.
    """
    user_message = "Strictly use only the references listed above and no other information, answer the following question: \n"
    user_message += f"Input Query: {input_question}"
    formatted_prompt = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    return formatted_prompt

def ner_movie_template(input_question):
    system_message = """Identify and list all the named entities present in the above-mentioned input question about movies and make SURE to NOT answer the question and list all the entities present in the question like movie name, person in the movie, actors in the movie, cast in the movie, etc. If there are no entities present in the question then in that case jsut output "No entities". The output format should be in the following format:
    Output: {
        "person_name": "name of a person or the character being referenced in the input question", 
        "movie_name": "name of a movie in the input question",
        "year": "if year is being referenced in the question then the requrired format is YYYY and date being referenced in the question, make sure that the format fo the ouput date is MM-DD-YYYY or if it a range YYYY to YYYY should the required format"
    }

    Make sure to include "Output:" before the dictionary. In the above format if any of the above mentioned categories are not present in the inpupt question then just ouput '' (Empty String) for that category.
    """
    user_message = "Strictly use only the references listed above and no other information, answer the following question: \n"
    user_message += f"Input Query: {input_question}"
    formatted_prompt = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    return formatted_prompt

def ner_finance_template(input_question):
    system_message = """Identify and list all the named entities present in the above-mentioned input question about finance and make SURE to NOT answer the question and list all the entities present in the question like company name, stock name, corporations, ticker symbols etc. If there are no entities present in the question then in that case jsut output "No entities". The output format should be in the following format:
    Output: {
        "compnay_name": "name of the company being referenced in the input question",
        "ticker_name": "name of the ticker (unique combination of letters and numbers used to identify a publicly traded security, like a stock, on a stock exchange) being referenced in the input question."
    }

    Make sure to include "Output:" before the dictionary. In the above format if any of the above mentioned categories are not present in the inpupt question then just ouput '' (Empty String) for that category.
    """
    user_message = "Strictly use only the references listed above and no other information, answer the following question: \n"
    user_message += f"Input Query: {input_question}"
    formatted_prompt = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    return formatted_prompt

def ner_open_template(input_question):
    system_message = """Identify and list all the named entities present in the above-mentioned input question and make SURE to NOT answer the question :
    Output: {
        "person_name": "name of a person being referenced in the input question",
        "location_name": "name of a location in the input question",
        "organization_name": "name of an organization in the input question",
        "product_name": "name of a product in the input question",
        "event_name": "name of an event in the input question"
    }

    Make sure to include "Output:" before the dictionary. In the above format if any of the above mentioned categories are not present in the inpupt question then just ouput '' (Empty String) for that category.
    """
    user_message = "Strictly use only the references listed above and no other information, answer the following question: \n"
    user_message += f"Input Query: {input_question}"
    formatted_prompt = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    return formatted_prompt

def ner_music_template(input_question):
    system_message = """Identify and list all the named entities present in the above-mentioned input question about music and make SURE to NOT answer the question and list all the entities present in the question like movie name, person in the song name, singer name, song writer name, band name, artisit name, etc. If there are no entities present in the question then in that case jsut output "No entities". The output format should be in the following format:
    Output: {
        "artist_name": "name of a person in the input question",
        "song_name": "name of a song in the input question",
        "date": "if year is being referenced in the question then the requrired format is YYYY and date being referenced in the question, make sure that the format fo the ouput date is MM-DD-YYYY or if it a range YYYY to YYYY should the required format",
        "band_nmae": "name of a band in the input question"
    }

    Make sure to include "Output:" before the dictionary. In the above format if any of the above mentioned categories are not present in the inpupt question then just ouput '' (Empty String) for that category.
    """
    user_message = "Strictly use only the references listed above and no other information, answer the following question: \n"
    user_message += f"Input Query: {question}"
    formatted_prompt = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    return formatted_prompt

def get_ner_prompt(domain, input_question):
    if domain == "music":
        ner_prompt = ner_music_template(input_question)
    elif domain == "movie":
        ner_prompt = ner_movies_template(input_question)
    elif domain == "sports":
        ner_prompt = ner_sports_template(input_question)
    elif domain == "finance":
        ner_prompt = ner_finance_template(input_question)
    elif domain == "open":
        ner_prompt = ner_open_template(input_question)
    else: 
        raise ValueError("Invalid domain passed")
    return ner_prompt
        
def classification_prompt_template(question):
    system_message = """You are a classifier and your task is to classify the given input query into classes. You will be given an input query and your task is to read the given query very carefully nad then classify the given query based on the following types of domains: 
    - "music"
    - "movie"
    - "sports"
    - "finance"
    - "open"
    A given query is cnosidered to belong to the domain "music" is the main theme or the main entity being talked about in the given input query belongs to or is related to music.
    Similarly, a given query is considered to belong to the domain "movie" is the main theme or the main entity being talked about in the given input query belongs to or is related to movies.
    Similarly, a given query is considered to belong to the domain "sports" is the main theme or the main entity being talked about in the given input query belongs to or is related to sports.
    Similarly, a given query is considered to belong to the domain "finanace" is the main theme or the main entity being talked about in the given input query belongs to or is related to finance.
    Similarly, a given query is considered to belong to the domain "movie" is the main theme or the main entity being talked about in the given input query does not belong to any of the above-mentioned classes, in that case it belongs to the open category.
    ### Examples: 
    1. Input Query: "where did the ceo of salesforce previously work?"
       Output: "finance"
    2. Input Query: "what album did the killers release in 2004, which included the songs "mr. brightside" and "jenny was a friend of mine"?"
       Output: "music"
    3. Input Query: "which movie won the oscar best visual effects in 2021?"
       Output: "movie"
    4. Input Query: "what's the name of nashville's hockey team?"
       Output: "sports"
    5. Input Query: "what are the countries that are located in southern africa?"
       Output: "open"
    6. Input Query: "what are the telephone area codes for the city of boise, idaho?"
       Output: "open"
    ### NOTE: Make sure the output is only a single word from only of the following words: "music", "movie", "sports", "finance" or "open". Do not output anything other than these three words. Be concise in the responses and do not generate irrelevant or extraneous text in the response.
    """
    user_message = "Strictly use only the references listed above and no other information, answer the following question: \n"
    user_message += f"Input Query: {question}"
    formatted_prompt = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    return formatted_prompt