import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV dataset."""
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["FTR"])  # only keep matches with results
    return df

def preprocess_full(df: pd.DataFrame):
    """Use all useful attributes from matches.csv."""
    # Encode target
    df['Target'] = df['FTR'].map({'H': 1, 'D': 0, 'A': -1})

    # Drop leakage columns (true results & final scores)
    drop_cols = ["FTR", "FTHG", "FTAG", "HTHG", "HTAG", "HTR"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Drop identifiers
    drop_text = ["Div", "Date", "Time", "Referee"]
    df = df.drop(columns=drop_text, errors="ignore")

    # Encode teams numerically
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    df["HomeTeam"] = le_home.fit_transform(df["HomeTeam"])
    df["AwayTeam"] = le_away.fit_transform(df["AwayTeam"])

    # Fill missing numeric values
    df = df.fillna(df.median(numeric_only=True))

    X = df.drop(columns=["Target"], errors="ignore")
    y = df["Target"]

    return X, y, le_home, le_away
