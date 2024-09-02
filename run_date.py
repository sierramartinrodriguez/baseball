import mlbstatsapi
import pandas as pd
import pybettor
import requests
from bs4 import BeautifulSoup as bs
import json
import builtins
type = builtins.type 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from sklearn import linear_model
import statistics
import matplotlib.pyplot as plt
import sys

# create mlb instance
mlb = mlbstatsapi.Mlb()

# set 2024 season param
params = {'season': 2024}

# set teams dict

# define win probability function
def win_prob(home_wp, away_wp):
    # free throw - win/win, loss/loss, win/loss, loss/win
    home_prob = home_wp * (1 - away_wp)
    away_prob = away_wp * (1 - home_wp)

    # normalize
    sum_of_pct = home_prob + away_prob
    normalized_home = round(float(home_prob / sum_of_pct), 3)

    # this only returns the normalized home, not with 0.025 adjustment
    return normalized_home 

# get a team's pythagorean win pct
def get_pythag_win_pct(rs, sp_rain, sp_avg_ip, team_rain, team_gp):
    # rs is just rs
    if sp_avg_ip < 0:
        team_ra = (team_rain * 9) * (team_gp)
    else:
        sp_ra_contribution = float(sp_avg_ip) * float(sp_rain)
        team_ra_contribution = float(9 - sp_avg_ip) * float(team_rain)
        total_ra_pg = sp_ra_contribution + team_ra_contribution
        team_ra = total_ra_pg * team_gp

    numerator = rs ** 1.83
    denom = (rs ** 1.83) + (team_ra ** 1.83)

    return round(float(numerator / denom), 3)

teams = {'Washington Nationals': 'WSH',
          'Toronto Blue Jays': 'TOR',
          'Texas Rangers': 'TEX', 
          'Tampa Bay Rays': 'TBR',
          'St. Louis Cardinals': 'STL',
          'Seattle Mariners': 'SEA', 
          'San Diego Padres': 'SDP',
          'San Francisco Giants': 'SFG',
          'Pittsburgh Pirates': 'PIT',
          'Philadelphia Phillies': 'PHI',
          'Oakland Athletics': 'OAK',
          'New York Mets': 'NYM',
          'New York Yankees': 'NYY',
          'Minnesota Twins': 'MIN',
          'Milwaukee Brewers': 'MIL', 
          'Miami Marlins': 'MIA', 
          'Los Angeles Angels': 'LAA',
          'Los Angeles Dodgers': 'LAD',
          'Kansas City Royals': 'KCR',
          'Houston Astros': 'HOU',
          'Detroit Tigers': 'DET',
          'Colorado Rockies': 'COL', 
          'Cleveland Guardians': 'CLE',
          'Cincinnati Reds': 'CIN',
          'Chicago White Sox': 'CWS',
          'Chicago Cubs': 'CHC',
          'Boston Red Sox': 'BOS',
          'Baltimore Orioles': 'BAL',
          'Atlanta Braves': 'ATL',
          'Arizona Diamondbacks': 'ARI'}

# util
def convert_to_replacement(col):
    if col == 'winPercentage_sp':
        return 0.294
    else:
        return -1
    
def filter_cols(df, size, score_present=True):    
    selected_cols_no_score = ['runs_sp', 'homeRuns_sp', 'strikeOuts_sp', 'baseOnBalls_sp', 'hits_sp',
                        'hitByPitch_sp', 'totalBases_sp', 'avg_sp', 'obp_sp', 'slg_sp', 'ops_sp',
                        'whip_sp', 'strikeoutWalkRatio_sp', 'runs_rp', 'homeRuns_rp', 'strikeOuts_rp',
                        'baseOnBalls_rp', 'hits_rp', 'hitByPitch_rp', 'totalBases_rp', 'avg_rp', 
                        'obp_rp', 'slg_rp', 'ops_rp', 'whip_rp', 'strikeoutWalkRatio_rp', 'homeRuns',
                        'strikeOuts', 'baseOnBalls', 'hits', 'hitByPitch', 'totalBases', 'avg',
                        'obp', 'slg', 'ops']
    selected_cols = selected_cols_no_score.copy()
    selected_cols.append('score')

    minimal_cols = ['runs_sp', 'homeRuns_sp', 'strikeOuts_sp', 'baseOnBalls_sp',
                    'avg_sp', 'obp_sp', 'slg_sp', 'ops_sp', 'whip_sp', 
                    'runs_rp','strikeOuts_rp',
                    'baseOnBalls_rp', 'hits_rp', 'avg_rp', 
                    'obp_rp', 'slg_rp', 'ops_rp', 'whip_rp', 'homeRuns',
                    'strikeOuts', 'baseOnBalls', 'hitByPitch', 'avg',
                    'obp', 'slg', 'ops', 'score'] 

    # medium_games_df = df[selected_cols]

    if size == "small":
        if score_present:
            return df[selected_cols]
        else:
            return df[selected_cols_no_score]

##### over under data ########
def get_stats_as_of_pitcher(start, end, p1_id, season):
    as_of_req = requests.get(f"https://statsapi.mlb.com/api/v1/people?personIds={p1_id}&hydrate=stats(group=[pitching],type=[byDateRange],startDate={start},endDate={end},season={season})")
    as_of_content = as_of_req.content
    as_of_content_decoded = as_of_content.decode('utf-8')
    as_of_data = json.loads(as_of_content_decoded)

    stat_dict_original = as_of_data['people'][0]['stats'][0]['splits'][0]['stat']
        
    stat_dict_original['hand_sp'] = as_of_data['people'][0]['pitchHand']['code']

    sp_normal = ['groundOuts', 'airOuts', 'runs', 'doubles',
                    'triples', 'homeRuns', 'strikeOuts', 'baseOnBalls',
                    'hits', 'hitByPitch', 'groundIntoDoublePlay', 
                    'earnedRuns', 'wildPitches', 'totalBases']
    sp_no_normal = ['avg', 'atBats', 'obp', 'slg', 'ops', 'era', 'whip',
                    'strikePercentage', 'groundOutsToAirouts',
                    'pitchesPerInning', 'strikeoutWalkRatio',
                    'strikeoutsPer9Inn', 'walksPer9Inn', 'hitsPer9Inn',
                    'runsScoredPer9', 'homeRunsPer9']
    
    return_stat_dict = {}

    norm_data_sp = {k: float(stat_dict_original[k] / stat_dict_original['battersFaced']) for k in sp_normal}
    non_norm_data_sp = {k: stat_dict_original[k] for k in sp_no_normal}

    for k, v in norm_data_sp.items():
        return_stat_dict[f"{k}_sp"] = v
    for k, v in non_norm_data_sp.items():
        return_stat_dict[f"{k}_sp"] = v

    return return_stat_dict

def get_stats_as_of_teams(start, end, team_id, season):
    # constants
    normalized = ['groundOuts', 'airOuts', 'doubles', 'triples',
                'homeRuns', 'strikeOuts', 'baseOnBalls', 'hits',
                'hitByPitch', 'groundIntoDoublePlay', 'totalBases',
                'rbi', 'leftOnBase', 'sacBunts', 'sacFlies']
    no_normal = ['avg', 'obp', 'slg', 'ops', 'plateAppearances', 'babip']
    
    # get one for versus lefty/righty, get
    as_of_req = requests.get(f" https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?group=hitting&season={season}&sportIds=1&stats=byDateRange&startDate={start}&endDate={end}")
    as_of_content = as_of_req.content
    as_of_content_decoded = as_of_content.decode('utf-8')
    as_of_data = json.loads(as_of_content_decoded)

    return_dict = {}

    stats = as_of_data['stats'][0]# gives all splits

    for split in stats['splits']:

        # return_dict['split['split']['code']'] = 
        all_split_data = split['stat']

        normalized_data = {k: float(all_split_data[k] / all_split_data['plateAppearances']) for k in normalized}
        non_normalized_data = {k: all_split_data[k] for k in no_normal}

        for k, v in normalized_data.items():
            return_dict[k] = v
        for k, v in non_normalized_data.items():
            return_dict[k] = v

    return return_dict

def get_stats_as_of_rp(start, end, team_id, season):

    rp_normal = ['groundOuts', 'airOuts', 'runs', 'doubles',
                    'triples', 'homeRuns', 'strikeOuts', 'baseOnBalls',
                    'hits', 'hitByPitch', 'groundIntoDoublePlay', 
                    'earnedRuns', 'wildPitches', 'totalBases']
    rp_no_normal = ['avg', 'atBats', 'obp', 'slg', 'ops',
                    'stolenBasePercentage', 'era', 'whip',
                    'strikePercentage', 'groundOutsToAirouts',
                    'pitchesPerInning', 'strikeoutWalkRatio',
                    'strikeoutsPer9Inn', 'walksPer9Inn', 'hitsPer9Inn',
                    'runsScoredPer9', 'homeRunsPer9']

    as_of_req = requests.get(f'https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?group=pitching&season={season}&stats=statSplits&sitCodes=rp')
    as_of_content = as_of_req.content
    as_of_content_decoded = as_of_content.decode('utf-8')
    as_of_data = json.loads(as_of_content_decoded)

    return_dict = {}

    stats = as_of_data['stats'][0]# gives all splits

    for split in stats['splits']:
        code = split['split']['code']

        # return_dict['split['split']['code']'] = 
        rp_stats = split['stat']

        norm_data_rp = {k: float(rp_stats[k] / rp_stats['battersFaced']) for k in rp_normal}
        non_norm_data_rp = {k: rp_stats[k] for k in rp_no_normal}

        for k, v in norm_data_rp.items():
            return_dict[f"{k}_{code}"] = v
        for k, v in non_norm_data_rp.items():
            return_dict[f"{k}_{code}"] = v

    return return_dict

def convert_to_replacement(col):
    if col == 'winPercentage_sp':
        return 0.294
    else:
        return -1

def get_date_game_info(games_start, games_end, past=False, d_start="2024-01-01", d_end="2024-08-26", not_started = False):

    oops_counter = 0

    req_shed = requests.get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&hydrate=probablePitcher&startDate={games_start}&endDate={games_end}')
    json_byte_sched = req_shed.content
    decoded_data_sched = json_byte_sched.decode('utf-8')
    sched_data = json.loads(decoded_data_sched)
    dates = sched_data['dates']

    two_team_games = []
    one_team_games = []
    labels = []

    for d in dates:
        season = d['date'][:4]

        print(f"Running {d['date']}")
        sched_games = d['games']

        for g in range (len(sched_games)):
            try: 
                game_0 = sched_games[g]

                if game_0['status']['codedGameState'] == 'D':
                    continue

                if not_started and (game_0['status']['codedGameState'] in ['I', 'F']):
                    continue

                if game_0['gameType'] != 'R':
                    print("Continuing, irregular game")
                    continue

                game_info_home = {}
                game_info_away = {}

                away = game_0['teams']['away']
                home = game_0['teams']['home']

                away_id = away['team']['id']
                home_id = home['team']['id']

                # -------- SP --------
                away_sp = {'a_sp_name' : away['probablePitcher']['fullName'],
                                        'a_sp_id' : away['probablePitcher']['id']}
                home_sp = {'h_sp_name' : home['probablePitcher']['fullName'],
                                        'h_sp_id' : home['probablePitcher']['id']}
                
                two_team_games.append({'home_id': home_id, 'away_id' : away_id, 'home' : home['team']['name'], 'away': away['team']['name'],
                                       'away_sp' : away_sp, 'home_sp': home_sp})
                
                away_sp_info = get_stats_as_of_pitcher(d_start, d_end, away_sp['a_sp_id'], season)
                home_sp_info = get_stats_as_of_pitcher(d_start, d_end, home_sp['h_sp_id'], season)

                for asp_k, asp_v in away_sp_info.items():
                    game_info_home[asp_k] = asp_v
                
                for hsp_k, hsp_v in home_sp_info.items():
                    game_info_away[hsp_k] = hsp_v

                # ---------- Hitting ------
                home_hitting_info = get_stats_as_of_teams(d_start, d_end, home_id, season)
                away_hitting_info = get_stats_as_of_teams(d_start, d_end, away_id, season)

                for hhk, hhv in home_hitting_info.items():
                    game_info_home[hhk] = hhv

                for ahk, ahv in away_hitting_info.items():
                    game_info_away[ahk] = ahv
                
                # --------- RP --------------
                home_rp_info = get_stats_as_of_rp(d_start, d_end, home_id, season)
                away_rp_info = get_stats_as_of_rp(d_start, d_end, away_id, season)

                for arpk, arpv in away_rp_info.items():
                    game_info_home[arpk] = arpv

                for hrpk, hrpv in home_rp_info.items():
                    game_info_away[hrpk] = hrpv


                # ------- Venue -----------------
                game_info_home['venue'] = game_0['venue']['name']
                game_info_away['venue'] = game_0['venue']['name']

                # get_score
                if past:
                    game_info_home['score'] = home['score']
                    game_info_away['score'] = away['score']
                else:
                    labels.append(home['team']['name'])
                    labels.append(away['team']['name'])

                home_good = True
                away_good = True

                for key, val in game_info_home.items():
                    if key not in ['hand_sp', 'venue']:
                        try:
                            game_info_home[key] = float(val)
                        except:
                            if convert_to_replacement(key) > 0:
                                game_info_home[key] = float(convert_to_replacement(key))
                            else:
                                print(f"Bad data, skipping. | {key} : {val}")
                                home_good = False

                for key, val in game_info_away.items():
                    if key not in ['hand_sp', 'venue']:
                        try:
                            game_info_away[key] = float(val)
                        except:
                            if convert_to_replacement(key) > 0:
                                game_info_away[key] = float(convert_to_replacement(key))
                            else:
                                print(f"Bad data, skipping. | {key} : {val}")
                                away_good = False

                if home_good:
                    one_team_games.append(game_info_home)
                
                if away_good:
                    one_team_games.append(game_info_away)
            except Exception as e:
                print(f"Error on {away['team']['name']} @ {home['team']['name']} | {e}")
                oops_counter += 1
                continue
            
    print("Counter", oops_counter)

    games_df = pd.DataFrame.from_dict(one_team_games)

    return games_df, labels, two_team_games

# for machine learning o/u:
def run_ridge(today_games, today_labels, retrain=False):
    if retrain:
        games = get_date_game_info("2024-08-10", "2024-08-26", True)
    else:
        games = pd.read_csv("one_team_games_0826.csv")

    full_games_selected = filter_cols(games, "small")

    target = full_games_selected['score']
    numeric = full_games_selected.loc[:, full_games_selected.columns != 'venue']
    features = numeric.loc[:, numeric.columns != 'score']

    X_train, X_test, y_train, y_test = train_test_split(features,target , 
                                    random_state=5,  
                                    test_size=0.2,  
                                    shuffle=True)
    
    ridge = linear_model.Ridge(alpha=0.5)
    ridge.fit(X_train, y_train)

    pred_tr = ridge.predict(X_train)
    pred_tst = ridge.predict(X_test)

    error_train = root_mean_squared_error(pred_tr, y_train)
    error_test = root_mean_squared_error(pred_tst, y_test)

    print("RMSE train:", error_train)
    print("RMSE test:", error_test)
    print("--------------------")

    diffs = []

    for i in range(len(y_test)):
        diffs.append(abs(y_test.iloc[i] - pred_tst[i]))

    print("Mean", statistics.mean(diffs))
    print("Median", statistics.median(diffs))

    print("------- Today's predictions: ")

    today_pred = ridge.predict(filter_cols(today_games, "small", False))

    i = 0

    print("-------------------")

    # set up return for one day
    return_dict = {}

    while i < (len(today_pred) - 1):
        home_pred = round(today_pred[i], 2)
        away_pred = round(today_pred[i + 1], 2)
        total_pred = round(home_pred + away_pred, 2)

        return_dict[f"{teams[today_labels[i + 1]]}at{teams[today_labels[i]]}"]= total_pred
 
        i += 2 
    
    print(return_dict)

    return return_dict

# -------------

# get team's stats for a given team ID
def get_team_stats(team_id):
    stats = ['season', 'seasonAdvanced']
    groups = ['pitching', 'hitting']
    params = {'season': 2024}

    team_stats = mlb.get_team_stats(team_id, stats=stats, groups=groups, **params)
    season_pitching = team_stats['pitching']['season']
    season_batting = team_stats['hitting']['season']

    for split in season_pitching.splits:
        for k, v in split.stat.__dict__.items():
            if k == 'runs':
                ra = v
            elif k == 'runsscoredper9':
                rain = round(float(v) / 9, 3)
            elif k == 'gamesplayed':
                gp = v
    
    for split in season_batting.splits:
        for k, v in split.stat.__dict__.items():
            if k == 'runs':
                rs = v
                break
    
    return {"rs" : rs, "ra" : ra, "rain" : rain, "gp" : gp}

# get player stats for pitcher:
def get_pitcher_stats(pitcher_name):
    try:
        player_id = mlb.get_people_id(pitcher_name)[0]
    except:
        print(f"Pitcher not found: {pitcher_name}")
        return {"rain" : 0, "ip" : 0, 'avg_ip' : -1}

    stats = ['season']
    groups = ['pitching']
    params = {'season': 2024}
    
    stat_dict = mlb.get_player_stats(player_id, stats=stats, groups=groups, **params)

    try:
        season_pitching_stat = stat_dict['pitching']['season']

        for split in season_pitching_stat.splits:
            for k, v in split.stat.__dict__.items():
                if k == 'runsscoredper9':
                    rain = round(float(v) / 9, 3)
                elif k == 'inningspitched':
                    ip = v
                elif k == 'gamesstarted':
                    if v == 0:
                        gs = -1
                        print(f"Using team ERA for: {pitcher_name}")
                    else:
                        gs = v
            
            if gs > 0:
                avg_ip = round(float(float(ip)) / gs, 3)
            else:
                avg_ip = -1
    except Exception as e:
        print(f"Unstable SP: {pitcher_name}. Debut?")
        return {"rain" : 0, "ip" : 0, 'avg_ip' : -1}
        

    if gs < 3:
        print(f"Unstable SP: {pitcher_name}. Games started: {gs}")
    elif avg_ip > 8:
        print(f"Unstable SP: {pitcher_name}. Check Relief stats")
        return {"rain" : 0, "ip" : 0, 'avg_ip' : -1}
    
    print(f"{pitcher_name} avg_ip: {avg_ip}")
    return {"rain" : rain, "ip" : ip, 'avg_ip' : avg_ip}

def run_game(g, games_df, over_under=None):
    skipFlag = False
        
    home_id = g['home_id']
    away_id = g['away_id']
    home = teams[g['home']]
    away = teams[g['away']]

    #if home == 'NYY':
    #    skipFlag = True
    if skipFlag:
        print(f"Skipping.... | Home: {home} | Away: {away}")
        return {'away' : 0, 'home' : 0,
            'awayFgvFD': 0, 'homeFGvFD' : 0,
            'away_yh' : 0, 'home_yh' : 0,
            'ou_dif' : 0, 'total' : 0,
            'away_wp': 0, 'home_wp' : 0,
            'away_fd': 0, 'home_fd': 0,
            'fg_away': 0, 'fg_home' : 0, 'juice' : 0}

    print(f"Home: {home} | Away: {away}")

    home_team_stats = get_team_stats(home_id)
    away_team_stats = get_team_stats(away_id)

    away_sp = g['away_sp']['a_sp_name']
    home_sp = g['home_sp']['h_sp_name']
    away_pitcher_stats = get_pitcher_stats(away_sp)
    home_pitcher_stats = get_pitcher_stats(home_sp)

    away_pythag = get_pythag_win_pct(away_team_stats['rs'],
                                    away_pitcher_stats['rain'],
                                    away_pitcher_stats['avg_ip'], 
                                    away_team_stats['rain'], 
                                    away_team_stats['gp'])
    home_pythag = get_pythag_win_pct(home_team_stats['rs'],
                                    home_pitcher_stats['rain'],
                                    home_pitcher_stats['avg_ip'], 
                                    home_team_stats['rain'], 
                                    home_team_stats['gp'])
                                    

    home_wp = round(win_prob(home_pythag, away_pythag) + 0.025, 3) 
    away_wp = 1 - home_wp

    away_fd_implied = round(pybettor.implied_prob(int(games_df.loc[games_df['away'] == away, 'fd_away'].iloc[0]), category="us")[0], 3)
    home_fd_implied = round(pybettor.implied_prob(int(games_df.loc[games_df['home'] == home, 'fd_home'].iloc[0]), category="us")[0], 3)
    juice = (away_fd_implied + home_fd_implied) - 1

    fg_away = games_df.loc[games_df['away'] == away, 'fg_wp_away'].iloc[0]
    fg_home = games_df.loc[games_df['home'] == home, 'fg_wp_home'].iloc[0]

    awayFGvFD = fg_away - away_fd_implied
    homeFGvFD = fg_home - home_fd_implied
    away_yh = away_wp - away_fd_implied
    home_yh = home_wp - home_fd_implied

    if over_under:
        try:
            total = over_under[f"{away}at{home}"]
            ou_diff = total - games_df.loc[games_df['home'] == home, 'ou_line'].iloc[0]
        except:
            total = games_df.loc[games_df['home'] == home, 'ou_line'].iloc[0]
            ou_diff = 0
    else:
        total = 0
        ou_diff = 0

    row = {'away' : away, 'home' : home,
            'awayFgvFD': awayFGvFD, 'homeFGvFD' : homeFGvFD,
            'away_yh' : away_yh, 'home_yh' : home_yh, 
            'ou_diff' : ou_diff, 'total' : total,
            'away_wp': away_wp, 'home_wp' : home_wp,
            'away_fd': away_fd_implied, 'home_fd': home_fd_implied,
            'fg_away': fg_away, 'fg_home' : fg_home, 'juice' : juice}
    
    return row

# return g obj if this is the correct game        
def find_game(dash_date, away_abbrev, home_abbrev):
    games = get_days_games(dash_date, not_started=False)

    # gives us home and away id's for a game:
    for g in games:
        api_home = g['home']
        api_away = g['away']

        if (teams[api_away] == away_abbrev) and (teams[api_home] == home_abbrev):
            return g

    raise Exception("Game not found")

def run_date(date, sg_df=None):
    # receive date as MMDD
    if sg_df is not None:
        games_df = sg_df
    else:
        games_df = pd.read_csv(f'games_2024{date}.csv')

    date = str(date)
    dash_date = f"2024-{date[:2]}-{date[2:]}"

    games_return = get_date_game_info(dash_date, dash_date, past=False, not_started = False)
    one_team_games = games_return[0]
    labels = games_return[1]
    two_team_games = games_return[2]

    rows = []

    if sg_df is not None:
        # i have to get g, which is just the one in which
        g = find_game(dash_date, sg_df['away'][0], sg_df['home'][0]) # TODO : Fix this one

        row = run_game(g, games_df)
        rows.append(row)
    else:
        print(one_team_games)
        over_unders = run_ridge(one_team_games, labels, retrain=False) # run_ridge(today_games, today_labels, retrain=False):

        rows = []

        for g in two_team_games:
            row = run_game(g, games_df, over_unders)
            rows.append(row)

    return pd.DataFrame(data=rows)

def run_single_game(date, away_abbrev, home_abbrev, away_fd, home_fd, ou_line, away_fg, home_fg, away_sp, home_sp):
    single_game_row = {'away' : away_abbrev, 'home': home_abbrev, 'fd_away' : float(away_fd), 'fd_home': float(home_fd), 'ou_line' : float(ou_line),
                       'fg_wp_away' : float(away_fg), 'fg_wp_home' : float(home_fg), 'away_sp' : away_sp, 'home_sp' : home_sp}
    single_game_df = pd.DataFrame(single_game_row, index = [0])
    return run_date(date, single_game_df)


def main():
    date = sys.argv[1]
    all_flag = sys.argv[2]

    if all_flag == 'all':
        print(f"Running games for {date[:2]}/{date[2:]}/2024")
        
        out = run_date(date)
    elif all_flag == 'single':
        print("Running single game")

        away_abbrev = sys.argv[3]
        home_abbrev = sys.argv[4]
        away_fd = sys.argv[5]
        home_fd = sys.argv[6]
        ou_line = sys.argv[7]
        away_fg = sys.argv[8]
        home_fg = sys.argv[9]
        away_sp = sys.argv[10]
        home_sp = sys.argv[11]

        out = run_single_game(date, away_abbrev, home_abbrev, away_fd, home_fd, ou_line, away_fg, home_fg, away_sp, home_sp)
    else:
        raise Exception("Game not found")
        

    out.to_csv('output.csv')
    print("Done!")

main()