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

print("1. Завантаження базових моделей (Wav2Vec 2.0 та Silero VAD)...")
processor = Wav2Vec2Processor.from_pretrained("anton-l/wav2vec2-large-xlsr-53-ukrainian")
wav2vec_model = Wav2Vec2Model.from_pretrained("anton-l/wav2vec2-large-xlsr-53-ukrainian")

# Завантажуємо VAD для ідеального обрізання тиші
vad_model, vad_utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = vad_utils


def get_features_from_audio(audio_array):
    """Отримує ознаки з 12-го шару Wav2Vec2 для конкретного аудіо-масиву"""
    speech = librosa.util.normalize(audio_array)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wav2vec_model(**inputs, output_hidden_states=True)
    features = outputs.hidden_states[12]
    return features.mean(dim=1).squeeze().numpy()


def process_file_with_vad_and_augment(file_path):
    """Чистить файл через VAD і створює нові тембри (штучних дітей)"""
    results = []
    
    # 1. Читаємо файл через librosa (щоб обійти помилку torchaudio)
    wav_array, sr = librosa.load(file_path, sr=16000)
    wav_tensor = torch.from_numpy(wav_array)
    
    # 2. VAD: Вирізаємо тишу
    speech_timestamps = get_speech_timestamps(wav_tensor, vad_model, sampling_rate=16000)
    
    if len(speech_timestamps) > 0:
        clean_wav = collect_chunks(speech_timestamps, wav_tensor).squeeze().numpy()
    else:
        clean_wav = wav_array # Якщо голос не знайдено, беремо як є
        
    # Додаємо базовий очищений вектор
    results.append(get_features_from_audio(clean_wav))
    
    # 3. АГРЕСИВНА АУГМЕНТАЦІЯ (Тільки для _orig.wav)
    # Створюємо штучних дикторів, щоб модель не прив'язувалася до тембру
    if "_orig.wav" in file_path.lower():
        # Імітація меншої дитини (вищий голос)
        pitch_up_1 = librosa.effects.pitch_shift(y=clean_wav, sr=16000, n_steps=2.5)
        pitch_up_2 = librosa.effects.pitch_shift(y=clean_wav, sr=16000, n_steps=4.0)
        
        # Імітація старшої дитини (нижчий голос)
        pitch_down_1 = librosa.effects.pitch_shift(y=clean_wav, sr=16000, n_steps=-2.0)
        
        results.append(get_features_from_audio(pitch_up_1))
        results.append(get_features_from_audio(pitch_up_2))
        results.append(get_features_from_audio(pitch_down_1))
        
    return results


DATA_DIR = "dataset_augmented"
X = []
y = []
groups = []

print("2. Екстракція векторів (Очищення VAD + Синтез нових тембрів)... Це займе час!")
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]

for i, filename in enumerate(files):
    parts = filename.split("_")
    if len(parts) >= 3:
        speaker_id = parts[0].lower()
        label = parts[1].lower()
        file_path = os.path.join(DATA_DIR, filename)

        try:
            # Отримуємо 1 вектор для звичайних файлів, або 4 вектори для оригіналів
            embs = process_file_with_vad_and_augment(file_path)
            
            for emb in embs:
                X.append(emb)
                y.append(label)
                groups.append(speaker_id)  # Всі штучні тембри належать тій самій групі (щоб не було витоку!)
        except Exception as e:
            pass

    if (i + 1) % 50 == 0:
        print(f"Оброблено {i + 1} / {len(files)} файлів (Зібрано {len(X)} векторів)...")

X = np.array(X)
groups = np.array(groups)

print("\n3. Правильне групування (Speaker-Independent Split)...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y_encoded, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
groups_train, groups_test = groups[train_idx], groups[test_idx]

print("--- ПЕРЕВІРКА ВИТОКУ ДАНИХ ---")
print(f"Диктори у НАВЧАННІ: {np.unique(groups_train)}")
print(f"Диктори у ТЕСТУВАННІ: {np.unique(groups_test)}")
print("------------------------------\n")

print("4. Тренування класифікатора Support Vector Machine...")
clf = SVC(kernel="linear", C=10.0, probability=True, random_state=42)
clf.fit(X_train, y_train)

print("5. Оцінка моделі...")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🏆 ОНОВЛЕНА ЧЕСНА точність на НЕЗНАЙОМИХ голосах: {acc * 100:.1f}%\n")

joblib.dump(clf, "fedienko_svm_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Готово! Найдосконалішу модель збережено.")