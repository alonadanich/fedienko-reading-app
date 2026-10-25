import os
import tempfile
import librosa
import torch
import numpy as np
import joblib
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import Wav2Vec2Processor, Wav2Vec2Model

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"

app = FastAPI(
    title="Fedienko Reading App API (Machine Learning Edition)", version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("1. Завантаження екстрактора Wav2Vec 2.0...")
model_name = "anton-l/wav2vec2-large-xlsr-53-ukrainian"
processor = Wav2Vec2Processor.from_pretrained(model_name)
wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)

print("2. Завантаження натренованої моделі SVM...")
clf = joblib.load("fedienko_svm_model.pkl")
le = joblib.load("label_encoder.pkl")
print("ML Сервер повністю готовий до роботи!")

# Словник для перекладу кирилиці з фронтенду у латинські мітки моделі
CYRILLIC_TO_LATIN = {
    "а": "a",
    "ак": "ak",
    "ал": "al",
    "ам": "am",
    "ан": "an",
    "ао": "ao",
    "ас": "as",
    "ау": "au",
    "і": "i",
    "ік": "ik",
    "іл": "il",
    "ім": "im",
    "ін": "in",
    "іс": "is",
    "к": "k",
    "ка": "ka",
    "кі": "ki",
    "ко": "ko",
    "ку": "ku",
    "ки": "ky",
    "л": "l",
    "ла": "la",
    "лі": "li",
    "ло": "lo",
    "лу": "lu",
    "ли": "ly",
    "м": "m",
    "ма": "ma",
    "мі": "mi",
    "мо": "mo",
    "му": "mu",
    "ми": "my",
    "н": "n",
    "на": "na",
    "ні": "ni",
    "но": "no",
    "ну": "nu",
    "ни": "ny",
    "о": "o",
    "ок": "ok",
    "ол": "ol",
    "ом": "om",
    "он": "on",
    "ос": "os",
    "с": "s",
    "са": "sa",
    "сі": "si",
    "со": "so",
    "су": "su",
    "си": "sy",
    "у": "u",
    "уа": "ua",
    "ук": "uk",
    "ул": "ul",
    "ум": "um",
    "ун": "un",
    "ус": "us",
    "и": "y",
    "ик": "yk",
    "ил": "yl",
    "им": "ym",
    "ин": "yn",
    "ис": "ys",
}


def get_embedding(file_path):
    audio_array, sr = librosa.load(file_path, sr=16000)
    audio_array, _ = librosa.effects.trim(audio_array, top_db=20)
    inputs = processor(
        audio_array, sampling_rate=16000, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        outputs = wav2vec_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


@app.post("/api/compare")
async def compare_audio(
    target_sound: str = Form(...),
    etalon_file: UploadFile = File(...),
    child_file: UploadFile = File(...),
):
    try:
        # Нам більше не потрібен файл мами для порівняння, але ми його приймаємо,
        # щоб не ламати логіку вашого фронтенду.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_child:
            tmp_child.write(await child_file.read())
            child_path = tmp_child.name

        # 1. Отримуємо вектор з голосу дитини
        emb = get_embedding(child_path)
        os.unlink(child_path)

        # 2. Передаємо вектор у нашу SVM модель
        probabilities = clf.predict_proba([emb])[0]

        # Беремо не 1, а 3 найімовірніші варіанти (Top-3 Accuracy)
        # np.argsort сортує індекси від найменшої ймовірності до найбільшої,
        # [-3:] бере три останні (найбільші), [::-1] перевертає їх за спаданням
        top_3_indices = np.argsort(probabilities)[-3:][::-1]

        # Отримуємо назви цих трьох звуків
        top_3_labels_latin = le.inverse_transform(top_3_indices)

        # 3. Перевірка результату
        target_cyrillic = target_sound.lower().strip()
        target_latin = CYRILLIC_TO_LATIN.get(target_cyrillic, target_cyrillic)

        # Якщо цільовий звук є хоча б в ТРІЙЦІ лідерів нейромережі
        if target_latin in top_3_labels_latin:
            verdict = "success"
            message = f"Чудово! ШІ розпізнав звук [{target_cyrillic.upper()}]"
            score = 100.0
        else:
            verdict = "retry"
            # Показуємо, що саме почула модель, для дебагу
            top_3_display = [lbl.upper() for lbl in top_3_labels_latin]
            message = f"Спробуй ще раз! Модель почула {top_3_display}"
            score = 0.0

        return {
            "target_sound": target_sound,
            "score": score,
            "threshold_applied": 0.0,
            "verdict": verdict,
            "message": message,
        }

    except Exception as e:
        return {"error": str(e)}
