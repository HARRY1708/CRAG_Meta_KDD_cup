from mock_api.apiwrapper.pycragapi import CRAG

api = CRAG()

def kg_context_retrieval(domain, entities):
    print("this is the value of domain: ", domain.strip().lower())
    print(domain.strip().lower() == "open")
    kg_results = []
    if domain.strip().lower() == "movie":
        if entities['person_name'] != '':
            kg_result = api.movie_get_person_info(str(entities['person_name']))
            kg_results.apppend(str(kg_result))
        if entities['movie_name'] != '':
            kg_result = api.movie_get_movie_info(str(['movie_name'])) 
            kg_results.apppend(str(kg_result))
        if entities['year'] != '':
            kg_result = api.movie_get_year_info(str(entities['year'])) 
            kg_results.apppend(str(kg_result))
        if entities['movie_id'] != '':
            kg_result = api.movie_get_movie_info_by_id(int(entities['movie_id']))
            kg_results.apppend(str(kg_result))
        if entities['person_id'] != '':
            kg_result = api.movie_get_person_info_by_id(int(entities['person_id']))
            kg_results.apppend(str(kg_result))
        
    elif domain.strip().lower() == "sports":
        if entities['team_name'] != '' and entities['date'] != '':
            kg_result = api.sports_soccer_get_games_on_date(str(entities['team_name']), str(entities['date'])) 
            kg_results.append(str(kg_result))
        if entities['team_name'] != '' and entities['date'] != '':
            kg_result = api.sports_nba_get_games_on_date(str(entities['team_name']), str(entities['date']))
            kg_results.append(str(kg_result))
        if entities['games_id'] != '':
            kg_result = api.sports_nba_get_play_by_play_data_by_game_ids(entities['games_id'])
            kg_results.append(str(kg_result))
        
        
    elif domain.strip().lower() == "open":
        print("this is the value of entities: ", entities['entity_name'])
        if entities['entity_name'] != '':
            if isinstance(entities['entity_name'], list):
                for entity in entities['entity_name']:
                    kg_result = api.open_search_entity_by_name(str(entity))
                    kg_results.append(str(kg_result))
            else: 
                kg_result = api.open_search_entity_by_name(str(entities['entity_name']))
                kg_results.append(str(kg_result))
        if entities['entity'] != '' or entities['entity_name'] != '':
            if isinstance(entities['entity_name'], list):
                print("this is a list!!!")
                for entity in entities['entity_name']:
                    kg_result = api.open_get_entity(str(entity))
                    kg_results.append(kg_result)
            elif isinstance(entities['entity'], list):
                for entity in entities['entity']:
                    kg_result = api.open_get_entity(str(entity))
                    kg_results.append(kg_result)
            else:
                kg_result1 = api.open_get_entity(str(entities['entity_name']))
                kg_result2 = api.open_get_entity(str(entities['entity']))
                kg_results.append(str(kg_result1))
                kg_results.append(str(kg_result2))
        
    elif domain.strip().lower() == "music":
        if entities['artist_name'] != '':
            kg_result = api.music_search_artist_entity_by_name(str(entities['artist_name'])) 
            kg_results.append(str(kg_result))
        if entities['song_name'] != '':
            kg_result = api.music_search_song_entity_by_name(str(entities['song_name'])) 
            kg_results.append(str(kg_result))
        if entities['rank'] != '' or entities['date'] != '':
            kg_result = api.music_get_billboard_rank_date(int(entities['rank']), str(entities['date'])) 
            kg_results.append(str(kg_result))
        if entities['date'] != '' and entities['attribute'] != '' and entities['song_name'] != '':
            kg_result = api.music_get_billboard_attributes(str(entities['date']), str(entities['attribute']), str(entities['song_name'])) 
            kg_results.append(str(kg_result))
        if entities['date'] != '':
            if len(entities['date']) == 10:
                date = entities['date'][6:]
            kg_result = api.music_grammy_get_best_artist_by_year(int(date)) 
            kg_results.append(str(kg_result))
        if entities['artist_name'] != '':
            kg_result = api.music_grammy_get_award_count_by_artist(str(entities['artist_name']))
            kg_results.append(str(kg_result))
        if entities['song_name'] != '':
            kg_result = api.music_grammy_get_award_count_by_song(str(entities['song_name'])) 
            kg_results.append(str(kg_result))
        if entities['date'] != '':
            if len(entities['date']) == 10:
                date = entities['date'][6:]
            kg_result = api.music_grammy_get_best_song_by_year(int(date)) 
            kg_results.append(str(kg_result))
        if entities['artist_name'] != '':
            kg_result = api.music_grammy_get_award_date_by_artist(str(entities['artist_name']))
            kg_results.append(str(kg_result))
        if entities['date'] != '':
            if len(entities['date']) == 10:
                date = entities['date'][6:]
            kg_result = api.music_grammy_get_best_album_by_year(int(date)) 
            kg_results.append(str(kg_result))
        if entities['artist_name'] != '':
            kg_result = api.music_get_artist_birth_place(str(entities['artist_name'])) 
            kg_results.append(str(kg_result))
        if entities['artist_name'] != '':
            kg_result = api.music_get_artist_birth_date(str(entities['artist_name'])) 
            kg_results.append(str(kg_result))
        if entities['band_name'] != '':
            kg_result = api.music_get_members(str( entities['band_name'])) 
            kg_results.append(str(kg_result))
        if entities['artist_name'] != '':
            kg_result = api.music_get_lifespan(str(entities['artist_name']))
            kg_results.append(str(kg_result))
        if entities['song_name'] != '':
            kg_result = api.music_get_song_author(str(entities['song_name']))
            kg_results.append(str(kg_result))
        if entities['song_name'] != '':
            kg_result = api.music_get_song_release_country(str(entities['song_name']))
            kg_results.append(str(kg_result))
        if entities['song_name'] != '':
            kg_result = api.music_get_song_release_date(str(entities['song_name'])) 
            kg_results.append(str(kg_result))
        if entities['artist_name'] != '':
            kg_result = api.music_get_artist_all_works(str(entities['artist_name']))
            kg_results.append(str(kg_result))
        kg_result = api.music_grammy_get_all_awarded_artists() 
        kg_results.append(str(kg_result))
        
    elif domain.strip().lower() == "finance":
        if entities['company_name'] != '':
            kg_result = api.finance_get_company_name(str(entities['company_name']))
            kg_results.append(str(kg_result))
        if entities['ticker_name'] != '':
            kg_result = api.finance_get_ticker_by_name(str(entities['ticker_name']))
            kg_results.append(str(kg_result))
        if entities['ticker_name'] != '':
            kg_result = api.finance_get_price_history(tr(entities['ticker_name'])) 
            kg_results.append(str(kg_result))
        if entities['ticker_name'] != '':
            kg_result = api.finance_get_detailed_price_history(str(entities['ticker_name'])) 
            kg_results.append(str(kg_result))
        if entities['ticker_name'] != '':
            kg_result = api.finance_get_dividends_history(str(entities['ticker_name'])) 
            kg_results.append(kg_result)
        if entities['ticker_name'] != '':
            kg_result = api.finance_get_market_capitalization(str(entities['ticker_name']))
            kg_results.append(kg_result)
        if entities['ticker_name'] != '':
            kg_result = api.finance_get_eps(str(entities['ticker_name']))
            kg_results.append(kg_result)
        if entities['ticker_name'] != '':
            kg_result = api.finance_get_pe_ratio(str(entities['ticker_name']))
            kg_results.append(kg_result)
        if entities['ticker_name'] != '':
            kg_result = api.finance_get_info(str(entities['ticker_name']))
            kg_results.append(kg_result)
        
    else:
        return "No results"
    
    return kg_results
