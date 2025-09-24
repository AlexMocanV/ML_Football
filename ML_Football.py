import os
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from preprocess import load_data, preprocess_full

class MatchNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=3):
        super(MatchNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

DATA_PATH = "data/matches.csv"
MODEL_LOGREG = "models/predictor_logreg.pkl"
MODEL_NN = "models/predictor_nn.pth"
ENCODERS_PATH = "models/encoders.pkl"
SCALER_PATH = "models/scaler.pkl"
os.makedirs("models", exist_ok=True)

def train_logreg():
    df = load_data(DATA_PATH)
    X, y, le_home, le_away = preprocess_full(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LogisticRegression(max_iter=5000, multi_class="multinomial")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Logistic Regression Accuracy: {acc:.2f}")
    joblib.dump(model, MODEL_LOGREG)
    joblib.dump((le_home, le_away), ENCODERS_PATH)

def train_nn():
    df = load_data(DATA_PATH)
    X, y, le_home, le_away = preprocess_full(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_nn = y.replace({-1:2})  # Away -> 2

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_nn, test_size=0.2)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.long)

    model = MatchNN(input_size=X_train.shape[1], hidden_size=128)
    # class weights
    class_counts = torch.tensor([len(y_nn[y_nn==1]), len(y_nn[y_nn==0]), len(y_nn[y_nn==2])], dtype=torch.float32)
    class_weights = 1.0 / class_counts
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(150):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        acc = (preds == y_test).float().mean()
        print(f"✅ Neural Network Accuracy: {acc:.2f}")

    torch.save(model.state_dict(), MODEL_NN)
    joblib.dump((le_home, le_away), ENCODERS_PATH)
    joblib.dump(scaler, SCALER_PATH)

def predict_match(home_team, away_team, model_type="logreg"):
    df = load_data(DATA_PATH)
    X, _, le_home, le_away = preprocess_full(df)

    if home_team not in le_home.classes_ or away_team not in le_away.classes_:
        print("⚠️ One of the teams not found in dataset.")
        return

    home_enc = le_home.transform([home_team])[0]
    away_enc = le_away.transform([away_team])[0]

    sample = X.median().to_dict()
    sample["HomeTeamEnc"] = home_enc
    sample["AwayTeamEnc"] = away_enc

    X_new = pd.DataFrame([sample])

    if model_type=="logreg":
        model = joblib.load(MODEL_LOGREG)
        probs = model.predict_proba(X_new)[0]
        classes = model.classes_
        mapping = {1:"Home Win",0:"Draw",-1:"Away Win"}

    elif model_type=="nn":
        scaler = joblib.load(SCALER_PATH)
        model = MatchNN(input_size=X_new.shape[1], hidden_size=128)
        model.load_state_dict(torch.load(MODEL_NN))
        model.eval()

        X_scaled = scaler.transform(X_new)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            probs = model(X_tensor)[0].numpy()
        classes = [1,0,2]
        mapping = {1:"Home Win",0:"Draw",2:"Away Win"}

    print(f"\n🔮 Prediction for {home_team} vs {away_team} ({model_type}):")
    for c,p in zip(classes, probs):
        print(f"{mapping[c]}: {p:.2%}")

if __name__=="__main__":
    print("1️⃣ Train Logistic Regression")
    print("2️⃣ Train Neural Network")
    print("3️⃣ Predict with Logistic Regression")
    print("4️⃣ Predict with Neural Network")
    choice = input("Choose option (1/2/3/4): ")
    if choice=="1":
        train_logreg()
    elif choice=="2":
        train_nn()
    elif choice=="3":
        home=input("Enter Home Team: ")
        away=input("Enter Away Team: ")
        predict_match(home, away,"logreg")
    elif choice=="4":
        home=input("Enter Home Team: ")
        away=input("Enter Away Team: ")
        predict_match(home, away,"nn")
    else:
        print("❌ Invalid choice")
