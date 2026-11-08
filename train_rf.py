import os
import librosa
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

print("1. Завантаження базової нейромережі (Wav2Vec 2.0)...")
processor = Wav2Vec2Processor.from_pretrained(
    "anton-l/wav2vec2-large-xlsr-53-ukrainian"
)
model = Wav2Vec2Model.from_pretrained("anton-l/wav2vec2-large-xlsr-53-ukrainian")


def get_embedding(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


DATA_DIR = "dataset_augmented"
X = []
y = []

print("2. Екстракція векторів з файлів... (Це може зайняти 5-10 хвилин)")
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]

for i, filename in enumerate(files):
    # Файли названі як "child_ma_fast.wav" або "mother_a_orig.wav"
    # Розбиваємо назву по "_" і беремо другий елемент (саму літеру/склад)
    parts = filename.split("_")
    if len(parts) >= 3:
        label = parts[1]
        file_path = os.path.join(DATA_DIR, filename)

        try:
            emb = get_embedding(file_path)
            X.append(emb)
            y.append(label)
        except Exception as e:
            print(f"Помилка з {filename}: {e}")

    if (i + 1) % 50 == 0:
        print(f"Оброблено {i + 1} / {len(files)} файлів...")

X = np.array(X)

print("3. Підготовка до машинного навчання...")
# LabelEncoder перетворює текстові мітки ('a', 'ma', 'k') у числа (0, 1, 2...)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Розбиваємо дані: 80% на навчання, 20% на іспит (тестування)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Навчання Random Forest...")
# n_estimators=100 означає створення 100 дерев
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)

print("Оцінка моделі...")
y_pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Точність Random Forest: {accuracy:.2f}%")
print(classification_report(y_test, y_pred))

joblib.dump(rf_clf, "fedienko_rf_model.pkl")
print("Модель збережено як fedienko_rf_model.pkl")
