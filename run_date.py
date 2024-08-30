import mlbstatsapi
import pandas as pd
import pybettor
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

# get the scheduled games for a given day. Default to only not started games
def get_days_games(d, not_started = False):
    schedule = mlb.get_scheduled_games_by_date(d)
    
    games = []

    for game in schedule:
        if not_started and (game.status.codedgamestate in ['I', 'F']):
            continue
        home_id = game.teams.home.team.id
        away_id = game.teams.away.team.id
        home = game.teams.home.team.name
        away = game.teams.away.team.name

        games.append({'home_id': home_id, 'away_id' : away_id, 'home' : home, 'away': away})
    
    return games

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

def run_game(g, games_df):
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
            'away_wp': 0, 'home_wp' : 0,
            'away_fd': 0, 'home_fd': 0,
            'fg_away': 0, 'fg_home' : 0, 'juice' : 0}

    print(f"Home: {home} | Away: {away}")

    home_team_stats = get_team_stats(home_id)
    away_team_stats = get_team_stats(away_id)

    away_sp = games_df.loc[games_df['away'] == away, 'away_sp'].iloc[0]
    home_sp = games_df.loc[games_df['home'] == home, 'home_sp'].iloc[0]
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

    row = {'away' : away, 'home' : home,
            'awayFgvFD': awayFGvFD, 'homeFGvFD' : homeFGvFD,
            'away_yh' : away_yh, 'home_yh' : home_yh,
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

    rows = []

    if sg_df is not None:
        # i have to get g, which is just the one in which
        g = find_game(dash_date, sg_df['away'][0], sg_df['home'][0])

        row = run_game(g, games_df)
        rows.append(row)
    else:
        games = get_days_games(dash_date, not_started=False)

        rows = []

        for g in games:
            row = run_game(g, games_df)
            rows.append(row)
    
    return pd.DataFrame(data=rows)

def run_single_game(date, away_abbrev, home_abbrev, away_fd, home_fd, away_fg, home_fg, away_sp, home_sp):
    single_game_row = {'away' : away_abbrev, 'home': home_abbrev, 'fd_away' : float(away_fd), 'fd_home': float(home_fd),
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
        away_fg = sys.argv[7]
        home_fg = sys.argv[8]
        away_sp = sys.argv[9]
        home_sp = sys.argv[10]

        out = run_single_game(date, away_abbrev, home_abbrev, away_fd, home_fd, away_fg, home_fg, away_sp, home_sp)
    else:
        raise Exception("Game not found")
        

    out.to_csv('output.csv')
    print("Done!")

main()