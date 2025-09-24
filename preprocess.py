import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["FTR"])  # keep only matches with result
    return df

def encode_teams(df):
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    df["HomeTeamEnc"] = le_home.fit_transform(df["HomeTeam"])
    df["AwayTeamEnc"] = le_away.fit_transform(df["AwayTeam"])
    return le_home, le_away

def compute_pre_match_stats(df, n_matches=5):
    df = df.copy()
    stats_cols = ["HomeGoalsScored","HomeGoalsConceded","HomeWinRate","HomeDrawRate","HomeLossRate",
                  "HomeShots","HomeCorners","AwayGoalsScored","AwayGoalsConceded","AwayWinRate",
                  "AwayDrawRate","AwayLossRate","AwayShots","AwayCorners"]
    for col in stats_cols:
        df[col] = 0.0

    for idx, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        # Home team last matches
        home_matches = df[(df["HomeTeam"]==home) | (df["AwayTeam"]==home)]
        home_prev = home_matches.iloc[:idx].tail(n_matches)
        if not home_prev.empty:
            hs = (home_prev["HS"].where(home_prev["HomeTeam"]==home, other=home_prev["AS"])).mean()
            hc = (home_prev["HC"].where(home_prev["HomeTeam"]==home, other=home_prev["AC"])).mean()
            hg = (home_prev["FTHG"].where(home_prev["HomeTeam"]==home, other=home_prev["FTAG"])).mean()
            ag = (home_prev["FTAG"].where(home_prev["HomeTeam"]==home, other=home_prev["FTHG"])).mean()
            wins = sum((home_prev["FTR"]=="H") & (home_prev["HomeTeam"]==home)) + sum((home_prev["FTR"]=="A") & (home_prev["AwayTeam"]==home))
            draws = sum(home_prev["FTR"]=="D")
            losses = n_matches - wins - draws
            df.at[idx,"HomeGoalsScored"]=hg
            df.at[idx,"HomeGoalsConceded"]=ag
            df.at[idx,"HomeWinRate"]=wins/n_matches
            df.at[idx,"HomeDrawRate"]=draws/n_matches
            df.at[idx,"HomeLossRate"]=losses/n_matches
            df.at[idx,"HomeShots"]=hs
            df.at[idx,"HomeCorners"]=hc

        # Away team last matches
        away_matches = df[(df["HomeTeam"]==away) | (df["AwayTeam"]==away)]
        away_prev = away_matches.iloc[:idx].tail(n_matches)
        if not away_prev.empty:
            ashots = (away_prev["HS"].where(away_prev["HomeTeam"]==away, other=away_prev["AS"])).mean()
            acorn = (away_prev["HC"].where(away_prev["HomeTeam"]==away, other=away_prev["AC"])).mean()
            ags = (away_prev["FTHG"].where(away_prev["HomeTeam"]==away, other=away_prev["FTAG"])).mean()
            hgs = (away_prev["FTAG"].where(away_prev["HomeTeam"]==away, other=away_prev["FTHG"])).mean()
            wins = sum((away_prev["FTR"]=="H") & (away_prev["HomeTeam"]==away)) + sum((away_prev["FTR"]=="A") & (away_prev["AwayTeam"]==away))
            draws = sum(away_prev["FTR"]=="D")
            losses = n_matches - wins - draws
            df.at[idx,"AwayGoalsScored"]=ags
            df.at[idx,"AwayGoalsConceded"]=hgs
            df.at[idx,"AwayWinRate"]=wins/n_matches
            df.at[idx,"AwayDrawRate"]=draws/n_matches
            df.at[idx,"AwayLossRate"]=losses/n_matches
            df.at[idx,"AwayShots"]=ashots
            df.at[idx,"AwayCorners"]=acorn

    return df

def compute_recent_form(df, n_home=3, n_away=3):
    df = df.copy()
    df["HomeWinStreak"]=0
    df["HomeGoalDiff"]=0
    df["AwayWinStreak"]=0
    df["AwayGoalDiff"]=0

    for idx, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        home_prev = df[(df["HomeTeam"]==home)].iloc[:idx].tail(n_home)
        if not home_prev.empty:
            df.at[idx,"HomeWinStreak"] = sum(home_prev["FTR"]=="H")
            df.at[idx,"HomeGoalDiff"] = (home_prev["FTHG"] - home_prev["FTAG"]).sum()

        away_prev = df[(df["AwayTeam"]==away)].iloc[:idx].tail(n_away)
        if not away_prev.empty:
            df.at[idx,"AwayWinStreak"] = sum(away_prev["FTR"]=="A")
            df.at[idx,"AwayGoalDiff"] = (away_prev["FTAG"] - away_prev["FTHG"]).sum()
    return df

def preprocess_full(df):
    le_home, le_away = encode_teams(df)
    df = compute_pre_match_stats(df, n_matches=5)
    df = compute_recent_form(df)

    # Fill bookmaker odds
    odds_cols = ["B365H","B365D","B365A"]
    for col in odds_cols:
        df[col] = df[col].fillna(df[col].median())

    df["Target"] = df["FTR"].map({"H":1,"D":0,"A":-1})

    # Drop unused string columns
    df = df.drop(columns=["FTR","HomeTeam","AwayTeam","Date","Time","Div","Referee","HTHG","HTAG","HTR"], errors="ignore")

    df = df.fillna(df.median(numeric_only=True))

    X = df.drop(columns=["Target"])
    y = df["Target"]

    return X, y, le_home, le_away
