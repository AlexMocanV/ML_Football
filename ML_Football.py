import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from preprocess import load_data, preprocess_full

DATA_PATH = os.path.join("data", "matches.csv")
MODEL_PATH = os.path.join("models", "predictor.pkl")
ENCODERS_PATH = os.path.join("models", "encoders.pkl")

def train_model():
    print("🔄 Loading and preprocessing data...")
    df = load_data(DATA_PATH)
    X, y, le_home, le_away = preprocess_full(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print(f"🧠 Training model on {X.shape[1]} features...")
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.2f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump((le_home, le_away), ENCODERS_PATH)
    print(f"💾 Model + encoders saved to {MODEL_PATH} and {ENCODERS_PATH}")

def predict_match(home_team: str, away_team: str):
    if not os.path.exists(MODEL_PATH):
        print("⚠️ Model not trained yet. Run training first.")
        return

    df = load_data(DATA_PATH)
    X, y, le_home, le_away = preprocess_full(df)

    model = joblib.load(MODEL_PATH)
    le_home, le_away = joblib.load(ENCODERS_PATH)

    # Check if team names exist
    if home_team not in le_home.classes_ or away_team not in le_away.classes_:
        print("⚠️ One of the teams not found in dataset.")
        return

    # Encode teams
    home_encoded = le_home.transform([home_team])[0]
    away_encoded = le_away.transform([away_team])[0]

    # Use median of other columns as placeholder (since we only know teams)
    sample = X.median().to_dict()
    sample["HomeTeam"] = home_encoded
    sample["AwayTeam"] = away_encoded
    X_new = pd.DataFrame([sample])

    # Predict probabilities
    probs = model.predict_proba(X_new)[0]
    classes = model.classes_
    mapping = {1: "Home Win", 0: "Draw", -1: "Away Win"}

    print(f"\n🔮 Prediction for {home_team} vs {away_team}:")
    for c, p in zip(classes, probs):
        print(f"{mapping[c]}: {p:.2%}")

if __name__ == "__main__":
    print("1️⃣ Train model")
    print("2️⃣ Predict match")
    choice = input("Choose option (1/2): ")

    if choice == "1":
        train_model()
    elif choice == "2":
        home = input("Enter Home Team: ")
        away = input("Enter Away Team: ")
        predict_match(home, away)
    else:
        print("❌ Invalid choice")
