import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(46, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=1)


class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(46, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=1)


class GatingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(46, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=1)



# Load preprocessing
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Load classical models
xgb_model = XGBClassifier()
xgb_model.load_model("models/xgb.json")

lgb_model = joblib.load("models/lgb.pkl")
rf_model = joblib.load("models/rf.pkl")

cat_model = CatBoostClassifier()
cat_model.load_model("models/cat.cbm")

lr_model = joblib.load("models/lr.pkl")
svm_model = joblib.load("models/svm.pkl")

# Load torch models
small_mlp = SmallMLP().to(device)
small_mlp.load_state_dict(torch.load("models/small_mlp.pt"))
small_mlp.eval()

deep_mlp = DeepMLP().to(device)
deep_mlp.load_state_dict(torch.load("models/deep_mlp.pt"))
deep_mlp.eval()

gating_model = GatingNetwork().to(device)
gating_model.load_state_dict(torch.load("models/gating.pt"))
gating_model.eval()



FEATURE_COLUMNS = [
"Dst Port","Flow Duration","Tot Fwd Pkts","Tot Bwd Pkts",
"TotLen Fwd Pkts","Fwd Pkt Len Max","Fwd Pkt Len Min","Fwd Pkt Len Mean",
"Bwd Pkt Len Max","Bwd Pkt Len Min","Bwd Pkt Len Mean",
"Flow Byts/s","Flow Pkts/s","Flow IAT Mean","Flow IAT Std",
"Flow IAT Max","Bwd IAT Tot","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Min",
"Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags",
"Pkt Len Var","FIN Flag Cnt","RST Flag Cnt","PSH Flag Cnt",
"ACK Flag Cnt","URG Flag Cnt","CWE Flag Count","Down/Up Ratio",
"Fwd Byts/b Avg","Fwd Pkts/b Avg","Fwd Blk Rate Avg",
"Bwd Byts/b Avg","Bwd Pkts/b Avg","Bwd Blk Rate Avg",
"Init Fwd Win Byts","Init Bwd Win Byts","Fwd Act Data Pkts",
"Fwd Seg Size Min","Active Mean","Active Std",
"Active Max","Idle Min"
]



def predict_single(input_dict):
    df = pd.DataFrame([input_dict])

    # Select required features
    X = df[FEATURE_COLUMNS].values

    # Scale
    X_scaled = scaler.transform(X)

    # Classical expert probabilities
    expert_outputs = []
    expert_outputs.append(xgb_model.predict_proba(X_scaled))
    expert_outputs.append(lgb_model.predict_proba(X_scaled))
    expert_outputs.append(rf_model.predict_proba(X_scaled))
    expert_outputs.append(cat_model.predict_proba(X_scaled))
    expert_outputs.append(lr_model.predict_proba(X_scaled))
    expert_outputs.append(svm_model.predict_proba(X_scaled))

    # Neural experts
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        expert_outputs.append(small_mlp(X_tensor).cpu().numpy())
        expert_outputs.append(deep_mlp(X_tensor).cpu().numpy())

    expert_outputs = np.stack(expert_outputs, axis=1)
    expert_outputs = torch.tensor(expert_outputs, dtype=torch.float32).to(device)

    # Gating
    with torch.no_grad():
        gates = gating_model(X_tensor)
        gates = gates.unsqueeze(2)
        final_output = torch.sum(gates * expert_outputs, dim=1)

        prediction = torch.argmax(final_output, dim=1).item()

    class_label = label_encoder.inverse_transform([prediction])[0]

    return class_label


if __name__ == "__main__":

    sample_input = {'Dst Port': 0, 'Protocol': 0, 'Timestamp': '14/02/2018 08:31:01', 'Flow Duration': 112641719, 'Tot Fwd Pkts': 3, 'Tot Bwd Pkts': 0, 'TotLen Fwd Pkts': 0, 'TotLen Bwd Pkts': 0, 'Fwd Pkt Len Max': 0, 'Fwd Pkt Len Min': 0, 'Fwd Pkt Len Mean': 0.0, 'Fwd Pkt Len Std': 0.0, 'Bwd Pkt Len Max': 0, 'Bwd Pkt Len Min': 0, 'Bwd Pkt Len Mean': 0.0, 'Bwd Pkt Len Std': 0.0, 'Flow Byts/s': 0.0, 'Flow Pkts/s': 0.0266331163, 'Flow IAT Mean': 56320859.5, 'Flow IAT Std': 139.3000358938, 'Flow IAT Max': 56320958, 'Flow IAT Min': 56320761, 'Fwd IAT Tot': 112641719, 'Fwd IAT Mean': 56320859.5, 'Fwd IAT Std': 139.3000358938, 'Fwd IAT Max': 56320958, 'Fwd IAT Min': 56320761, 'Bwd IAT Tot': 0, 'Bwd IAT Mean': 0.0, 'Bwd IAT Std': 0.0, 'Bwd IAT Max': 0, 'Bwd IAT Min': 0, 'Fwd PSH Flags': 0, 'Bwd PSH Flags': 0, 'Fwd URG Flags': 0, 'Bwd URG Flags': 0, 'Fwd Header Len': 0, 'Bwd Header Len': 0, 'Fwd Pkts/s': 0.0266331163, 'Bwd Pkts/s': 0.0, 'Pkt Len Min': 0, 'Pkt Len Max': 0, 'Pkt Len Mean': 0.0, 'Pkt Len Std': 0.0, 'Pkt Len Var': 0.0, 'FIN Flag Cnt': 0, 'SYN Flag Cnt': 0, 'RST Flag Cnt': 0, 'PSH Flag Cnt': 0, 'ACK Flag Cnt': 0, 'URG Flag Cnt': 0, 'CWE Flag Count': 0, 'ECE Flag Cnt': 0, 'Down/Up Ratio': 0, 'Pkt Size Avg': 0.0, 'Fwd Seg Size Avg': 0.0, 'Bwd Seg Size Avg': 0.0, 'Fwd Byts/b Avg': 0, 'Fwd Pkts/b Avg': 0, 'Fwd Blk Rate Avg': 0, 'Bwd Byts/b Avg': 0, 'Bwd Pkts/b Avg': 0, 'Bwd Blk Rate Avg': 0, 'Subflow Fwd Pkts': 3, 'Subflow Fwd Byts': 0, 'Subflow Bwd Pkts': 0, 'Subflow Bwd Byts': 0, 'Init Fwd Win Byts': -1, 'Init Bwd Win Byts': -1, 'Fwd Act Data Pkts': 0, 'Fwd Seg Size Min': 0, 'Active Mean': 0.0, 'Active Std': 0.0, 'Active Max': 0, 'Active Min': 0, 'Idle Mean': 56320859.5, 'Idle Std': 139.3000358938, 'Idle Max': 56320958, 'Idle Min': 56320761, 'Label': 'Benign'}

    result = predict_single(sample_input)
    print("Final Prediction:", result)


