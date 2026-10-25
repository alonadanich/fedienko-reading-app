import os
import librosa
import torch
import numpy as np
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from scipy.spatial.distance import cosine

# Примусово вимикаємо токени для безпечного завантаження 
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"

print("Завантаження моделі Wav2Vec 2.0 (у режимі вилучення ознак)...")
model_name = "anton-l/wav2vec2-large-xlsr-53-ukrainian"

processor = Wav2Vec2Processor.from_pretrained(model_name)
# УВАГА: Використовуємо базову модель (Wav2Vec2Model), а не ForCTC!
# Нам потрібен не текст, а глибокі акустичні вектори (embeddings).
model = Wav2Vec2Model.from_pretrained(model_name)
print("Модель успішно завантажено!\n")


def get_wav2vec_embedding(file_path):
    """Витягує нейромережевий вектор звуку за допомогою PyTorch та Wav2Vec"""
    audio_array, sr = librosa.load(file_path, sr=16000)

    # Відрізаємо тишу, щоб порівнювати лише корисний сигнал
    audio_array, _ = librosa.effects.trim(audio_array, top_db=20)

    inputs = processor(
        audio_array, sampling_rate=16000, return_tensors="pt", padding=True
    )

    with torch.no_grad():
        # Отримуємо приховані стани (hidden states) нейромережі
        outputs = model(**inputs)

    # Беремо останній шар нейромережі і усереднюємо його по часу
    # Це створює єдиний багатовимірний "цифровий відбиток" звуку
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings


def compare_audio_embeddings(file_etalon, file_test):
    """Порівнює два ембединги за косинусною подібністю"""
    emb1 = get_wav2vec_embedding(file_etalon)
    emb2 = get_wav2vec_embedding(file_test)

    # Косинусна подібність
    similarity = 1 - cosine(emb1, emb2)
    return max(0.0, similarity * 100.0)


# ==========================================
# ПАКЕТНЕ ТЕСТУВАННЯ (WAV2VEC EMBEDDINGS)
# ==========================================

print("Починаємо порівняння нейромережевих векторів...\n")

test_pairs = [
    # 1. Позитивні тести (однакові звуки)
    {"звук": "а", "etalon": "test_audio/a_.wav", "child": "test_audio/a.wav"},
    {"звук": "а", "etalon": "test_audio/a_.wav", "child": "test_audio/a1.wav"},
    {"звук": "м", "etalon": "test_audio/m.wav", "child": "test_audio/m1.wav"},
    {"звук": "аy", "etalon": "test_audio/au_.wav", "child": "test_audio/au.wav"},
    {"звук": "аy", "etalon": "test_audio/au_.wav", "child": "test_audio/au1.wav"},
    {"звук": "му", "etalon": "test_audio/mu_.wav", "child": "test_audio/mu.wav"},
    {"звук": "му", "etalon": "test_audio/mu_.wav", "child": "test_audio/mu1.wav"},
    {"звук": "p", "etalon": "test_audio/r.wav", "child": "test_audio/r1.wav"},
    {"звук": "o", "etalon": "test_audio/o.wav", "child": "test_audio/o1.wav"},
    {"звук": "у", "etalon": "test_audio/u_.wav", "child": "test_audio/u.wav"},
    {"звук": "ма", "etalon": "test_audio/ma_.wav", "child": "test_audio/ma.wav"},
    {"звук": "yм", "etalon": "test_audio/um_.wav", "child": "test_audio/um.wav"},
    {"звук": "yм", "etalon": "test_audio/um_.wav", "child": "test_audio/um1.wav"},
    # 2. Негативні тести (Абсолютно різні звуки)
    {
        "звук": "Помилка (А vs У)",
        "etalon": "test_audio/a_.wav",
        "child": "test_audio/u.wav",
    },
    {
        "звук": "Помилка (А vs М)",
        "etalon": "test_audio/a_.wav",
        "child": "test_audio/m.wav",
    },
    {
        "звук": "Помилка (А vs O)",
        "etalon": "test_audio/a_.wav",
        "child": "test_audio/o.wav",
    },
    {
        "звук": "Помилка (M vs N)",
        "etalon": "test_audio/m.wav",
        "child": "test_audio/n.wav",
    },    
        
]

results = []

for pair in test_pairs:
    sound_name = pair["звук"]
    etalon_path = pair["etalon"]
    child_path = pair["child"]

    try:
        score = compare_audio_embeddings(etalon_path, child_path)

        print(f"[{sound_name}] | Схожість Wav2Vec векторів: {score:.1f}%")

        results.append({"Звук": sound_name, "Схожість Wav2Vec (%)": round(score, 1)})

    except Exception as e:
        print(f"Помилка [{sound_name}]: {e}")

if results:
    df = pd.DataFrame(results)
    df.to_csv("wav2vec_embeddings_results.csv", index=False, encoding="utf-8-sig")
    print(f"\nРезультати збережено у: wav2vec_embeddings_results.csv")
