import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

data_dict = joblib.load("features_layers_cache.pkl")
X = data_dict["singles"]["X"][8]
y = data_dict["singles"]["y"]
groups = data_dict["singles"]["groups"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

gkf = GroupKFold(n_splits=5)
scores = []
for tr_idx, te_idx in gkf.split(X, y_enc, groups):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X[tr_idx])
    X_te = sc.transform(X[te_idx])
    clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=42)
    clf.fit(X_tr, y_enc[tr_idx])
    scores.append(accuracy_score(y_enc[te_idx], clf.predict(X_te)))

print(f"Чесна точність SINGLES (Шар 8 + MLP): {np.mean(scores) * 100:.2f}%")

final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)
final_mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=42)
final_mlp.fit(X_scaled, y_enc)

joblib.dump(final_mlp, "model_mlp_singles.pkl")
joblib.dump(final_scaler, "scaler_singles.pkl")
joblib.dump(le, "le_singles.pkl")
print("Файли для одиничних звуків збережено!")