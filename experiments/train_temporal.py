import os
import librosa
import torch
import numpy as np
import warnings

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

print("1. Завантаження Wav2Vec 2.0...")
processor = Wav2Vec2Processor.from_pretrained("anton-l/wav2vec2-large-xlsr-53-ukrainian")
wav2vec_model = Wav2Vec2Model.from_pretrained("anton-l/wav2vec2-large-xlsr-53-ukrainian")

def get_temporal_features(file_path):
    """Розбиває звук на 3 частини (початок, середина, кінець) для розуміння порядку звуків"""
    speech, _ = librosa.load(file_path, sr=16000)
    speech, _ = librosa.effects.trim(speech, top_db=20)
    speech = librosa.util.normalize(speech)
    
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wav2vec_model(**inputs, output_hidden_states=True)
    
    # Беремо 12-й шар. Розмірність: [1, кількість_кадрів, 1024]
    features = outputs.hidden_states[12].squeeze(0) # Прибираємо зайвий вимір
    
    # Якщо звук занадто короткий, просто дублюємо кадри
    if features.shape[0] < 3:
        features = features.repeat(3, 1)
        
    # РОЗБИВАЄМО НА 3 БЛОКИ: Початок, Середина, Кінець
    chunks = torch.tensor_split(features, 3, dim=0)
    
    # Усереднюємо кожен блок окремо і склеюємо (1024 + 1024 + 1024 = 3072 ознаки)
    combined_emb = torch.cat([chunk.mean(dim=0) for chunk in chunks])
    return combined_emb.numpy()

DATA_DIR = "dataset_augmented"
X, y, groups = [], [], []

print("2. Екстракція часових ознак (Temporal Pooling)... Це займе 5-10 хвилин.")
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]

for i, filename in enumerate(files):
    parts = filename.split("_")
    if len(parts) >= 3:
        speaker_id = parts[0].lower()
        label = parts[1].lower()
        file_path = os.path.join(DATA_DIR, filename)
        
        try:
            emb = get_temporal_features(file_path)
            X.append(emb)
            y.append(label)
            groups.append(speaker_id)
        except Exception as e:
            pass
            
    if (i + 1) % 50 == 0:
        print(f"Оброблено {i + 1} / {len(files)} файлів...")

X, groups = np.array(X), np.array(groups)

print("\n3. Групування (Speaker-Independent Split) та Масштабування...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y_encoded, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("4. Тренування багатошарової нейромережі (MLP)...")
# Нейромережа з двома прихованими шарами (512 та 256 нейронів) з регуляризацією
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(512, 256), 
    activation='relu', 
    solver='adam', 
    alpha=0.01,           # L2 регуляризація проти перенавчання
    batch_size=64, 
    learning_rate='adaptive', 
    max_iter=300, 
    random_state=42,
    early_stopping=True   # Зупиниться, якщо точність перестане рости
)
mlp_clf.fit(X_train_scaled, y_train)

print("5. Оцінка моделі...")
y_pred = mlp_clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print(f"🏆 ЧЕСНА ТОЧНІСТЬ З ЧАСОВИМ РОЗДІЛЕННЯМ (Temporal Pooling): {acc * 100:.2f}%")
print("="*50)

joblib.dump(mlp_clf, "fedienko_mlp_temporal.pkl")
joblib.dump(scaler, "fedienko_scaler_temporal.pkl")
joblib.dump(le, "label_encoder_temporal.pkl")
print("Модель збережено!")