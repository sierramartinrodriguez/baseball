import pandas as pd
import pybettor
import sys

class MLB:
    def __init__(self, date="na"):
        if date != "na":
            self.standings_path = f"standings_2024{date}.csv"
            self.games_path = f"games_2024{date}.csv"
    
        # dict mapping abbreviations to bref teams
        self.teams = {'Washington Nationals': 'WSH',
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

    # gives a base for two teams
    def _win_prob(self, home_wp, away_wp):
        # free throw - win/win, loss/loss, win/loss, loss/win
        home_prob = home_wp * (1 - away_wp)
        away_prob = away_wp * (1 - home_wp)

        # normalize
        sum_of_pct = home_prob + away_prob
        normalized_home = round(float(home_prob / sum_of_pct), 3)
        #normalized_away = round(float(away_prob / sum_of_pct), 3)

        # return home probability over away probability
        return normalized_home
    
    # we are going to be expecting a csv with standings and:
    # team name, wins, losses, pythag. win/loss
    # Tm,W,L,pythWL
    def _process_team_standings(self):
        dtype = {"Tm": str, "W": int, "L" : int, "pythWL": str, "ERA": float}
        self.standings = pd.read_csv(self.standings_path, dtype=dtype)

        self.standings['tm'] = self.standings.apply(lambda row: self.teams[row.Tm], axis=1)
        self.standings['xW'] = self.standings.apply(lambda row: row.pythWL[:2], axis=1)
        self.standings['xL'] = self.standings.apply(lambda row: row.pythWL[-2:], axis=1)
        self.standings['GP'] = self.standings.apply(lambda row: row.W + row.L, axis=1)
        self.standings['wp'] = self.standings.apply(lambda row: round(float(row.W / row.GP), 3), axis=1)
        self.standings['xWP'] = self.standings.apply(lambda row: round(float(int(row.xW) / row.GP), 3), axis=1)
        self.standings = self.standings[['tm', 'GP', 'W', 'L', 'wp', 'xW', 'xL', 'xWP', 'ERA']]

    def _calc_pitcher_rd(self, pitcher_era, team_era, pitcher_ippg=5.33):
        if pitcher_era == "na":
            return 0
        
        team_ra_per_inning = float(team_era) / 9
        pitcher_ra_per_inning = float(pitcher_era) / 9
        team_exp_ra = pitcher_ippg * team_ra_per_inning
        pitcher_exp_ra = pitcher_ippg * pitcher_ra_per_inning
        diff = pitcher_exp_ra - team_exp_ra

        return diff # negative is good, positive bad
    
    def _process_games(self):
        self.xwp_dict = dict(zip(self.standings['tm'], self.standings['xWP']))
        self.era_dict = dict(zip(self.standings['tm'], self.standings['ERA']))

        games = pd.read_csv(self.games_path)
        games['away_xWP'] = games.apply(lambda row: self.xwp_dict[row.away], axis=1)
        games['home_xWP'] = games.apply(lambda row: self.xwp_dict[row.home], axis=1)
        games['homeProb'] = games.apply(lambda row: round(self._win_prob(row.home_xWP, row.away_xWP) + 0.025, 3), axis=1)
        games['awayProb'] = games.apply(lambda row: (1 - row.homeProb), axis=1)
        games['awayFDImplied'] = games.apply(lambda row: round(pybettor.implied_prob(row.fd_away, category="us")[0], 3), axis=1)
        games['homeFDImplied'] = games.apply(lambda row: round(pybettor.implied_prob(row.fd_home, category="us")[0], 3), axis=1)
        games['awayAdvTVvFD'] = games.apply(lambda row: row.awayProb - row.awayFDImplied, axis=1)
        games['homeAdvTVvFD'] = games.apply(lambda row: row.homeProb - row.homeFDImplied, axis=1)
        games['awayAdvFGvFD'] = games.apply(lambda row: row.fg_wp_away - row.awayFDImplied, axis=1)
        games['homeAdvFGvFD'] = games.apply(lambda row: row.fg_wp_home - row.homeFDImplied, axis=1)
        games['away_pitcher_rd'] = games.apply(lambda row: self._calc_pitcher_rd(row.away_sp_era, self.era_dict[row.away]), axis=1)
        games['home_pitcher_rd'] = games.apply(lambda row: self._calc_pitcher_rd(row.home_sp_era, self.era_dict[row.home]), axis=1)

        self.games = games
    
    def _get_insights(self):
        games_diff = self.games[['away', 'home', 'awayAdvTVvFD', 'homeAdvTVvFD', 'awayAdvFGvFD', 'homeAdvFGvFD', 'away_pitcher_rd', 'home_pitcher_rd' ]]
        games_diff = games_diff.round(3)
        return games_diff
    
    def run_all_games(self):
        self._process_team_standings()
        self._process_games()
        return self._get_insights()

def main():
    inp = sys.argv[1]
    print(f"Running games for {inp[:2]}/{inp[2:]}/2024")
    league_state = MLB(inp)
    out = league_state.run_all_games()
    out.to_csv('output.csv')
    print(out)

main()