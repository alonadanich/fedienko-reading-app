import joblib
import os
import librosa
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
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
print("Стиснення векторів з 1024 до 128 ознак (PCA)...")
pca = PCA(n_components=128, random_state=42)

# Навчаємо PCA на тренувальних даних і одразу стискаємо їх
X_train_pca = pca.fit_transform(X_train)
# Тестові дані тільки стискаємо (не навчаємо на них)
X_test_pca = pca.transform(X_test)

print(f"Розмір вектора до: {X_train.shape[1]}, після: {X_train_pca.shape[1]}")

print("Навчання SVM на стиснутих даних...")
svm_pca_clf = SVC(kernel="rbf", probability=True, random_state=42)
svm_pca_clf.fit(X_train_pca, y_train)

print("Оцінка моделі...")
y_pred = svm_pca_clf.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Точність SVM + PCA: {accuracy:.2f}%")

# Зберігаємо модель та сам оптимізатор PCA (він потрібен для UI)
joblib.dump(svm_pca_clf, "fedienko_svm_pca_model.pkl")
joblib.dump(pca, "pca_transformer.pkl")
print("Моделі збережено!")
