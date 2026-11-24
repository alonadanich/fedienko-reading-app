import os
import librosa
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings("ignore")

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

print("1. Завантаження базової нейромережі (Wav2Vec 2.0)...")
processor = Wav2Vec2Processor.from_pretrained("anton-l/wav2vec2-large-xlsr-53-ukrainian")
model = Wav2Vec2Model.from_pretrained("anton-l/wav2vec2-large-xlsr-53-ukrainian")

def get_embedding(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Використовуємо наш найкращий 12-й шар!
    features = outputs.hidden_states[12]
    return features.mean(dim=1).squeeze().numpy()


DATA_DIR = "dataset_augmented"
X = []
y = []
groups = []  # <--- ТУТ МИ ЗБЕРІГАТИМЕМО ID ДИТИНИ

print("2. Екстракція векторів з файлів... (Це може зайняти 5-10 хвилин)")
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]

for i, filename in enumerate(files):
    # Приклад назви: "child1_a_noisy.wav"
    parts = filename.split("_")
    if len(parts) >= 3:
        speaker_id = parts[0].lower() # Беремо ідентифікатор дитини (напр. 'child1')
        label = parts[1].lower()      # Беремо літеру/склад (напр. 'a')
        
        file_path = os.path.join(DATA_DIR, filename)

        try:
            emb = get_embedding(file_path)
            X.append(emb)
            y.append(label)
            groups.append(speaker_id)
        except Exception as e:
            pass

    if (i + 1) % 50 == 0:
        print(f"Оброблено {i + 1} / {len(files)} файлів...")

X = np.array(X)
groups = np.array(groups)

print("3. Правильне групування (Speaker-Independent Split)...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# РОБИМО МАГІЮ: Розбиваємо дані так, щоб спікери не перетиналися (80% дітей на навчання, 20% на тест)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y_encoded, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
groups_train, groups_test = groups[train_idx], groups[test_idx]

print("\n--- ПЕРЕВІРКА ВИТОКУ ДАНИХ ---")
print(f"Діти, які потрапили у навчання: {np.unique(groups_train)}")
print(f"Діти, які потрапили у ТЕСТУВАННЯ: {np.unique(groups_test)}")
print("------------------------------\n")

print("4. Тренування класифікатора Support Vector Machine...")
clf = SVC(kernel="linear", C=10.0, probability=True, random_state=42)
clf.fit(X_train, y_train)

print("5. Оцінка моделі...")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🏆 ЧЕСНА точність на НЕЗНАЙОМИХ дітях: {acc * 100:.1f}%\n")

joblib.dump(clf, "fedienko_svm_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Готово! Нова, чесна модель збережена.")