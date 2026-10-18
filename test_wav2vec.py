import os
import librosa
import torch
import pandas as pd
import re
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


# 1. ФУНКЦІЯ ОЧИЩЕННЯ
def normalize_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"([а-яіїєґ])\1+", r"\1", text)
    return text.strip()


print("Завантаження моделі Wav2Vec 2.0...")

# Використовуємо 100% публічну, еталонну українську модель
model_name = "anton-l/wav2vec2-large-xlsr-53-ukrainian"

processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
print("Модель Wav2Vec 2.0 завантажено успішно!\n")

folder_path = "test_audio"
results = []

# 3. ЦИКЛ ОБРОБКИ ФАЙЛІВ
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)

        try:
            audio_array, sampling_rate = librosa.load(file_path, sr=16000)

            # Wav2Vec2 приймає звук інакше, ніж Whisper
            inputs = processor(
                audio_array, sampling_rate=16000, return_tensors="pt", padding=True
            )

            with torch.no_grad():
                logits = model(inputs.input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]

            cleaned_transcription = normalize_text(transcription)

            results.append(
                {
                    "Назва файлу": filename,
                    "Розпізнаний текст Wav2Vec": transcription,
                    "Очищений текст": cleaned_transcription,
                }
            )

        except Exception as e:
            print(f"Помилка при обробці файлу {filename}: {e}")

# 4. ПІДРАХУНОК МЕТРИК ТА АНАЛІЗ
df = pd.DataFrame(results)

# Наш еталонний датасет
expected_mapping = {
    "a.wav": "а",
    "a1.wav": "а",
    "am.wav": "ам",
    "an1.wav": "ан",
    "au.wav": "ау",
    "au1.wav": "ау",
    "m.wav": "м",
    "m1.wav": "м",
    "ma.wav": "ма",
    "ma1.wav": "ма",
    "mama.wav": "мама",
    "mama1.wav": "мама",
    "mamo.wav": "мамо",
    "mamu.wav": "маму",
    "mamu1.wav": "маму",
    "mu.wav": "му",
    "mu1.wav": "му",
    "n.wav": "н",
    "n1.wav": "н",
    "o.wav": "о",
    "o1.wav": "о",
    "r.wav": "р",
    "r1.wav": "р",
    "u.wav": "у",
    "u1.wav": "у",
    "um.wav": "ум",
    "um1.wav": "ум",
}

df["Очікуваний текст"] = df["Назва файлу"].map(expected_mapping)
df["Точний збіг"] = df["Очищений текст"] == df["Очікуваний текст"]

accuracy = df["Точний збіг"].mean() * 100

print("=== НАУКОВИЙ ЗВІТ (Wav2Vec 2.0) ===")
print(f"Точність моделі Wav2Vec 2.0: {accuracy:.1f}%\n")

errors_df = df[df["Точний збіг"] == False]
print("=== АНАЛІЗ ПОМИЛОК WAV2VEC 2.0 ===")
print(errors_df[["Назва файлу", "Очікуваний текст", "Очищений текст"]])

df.to_csv("wav2vec_final_evaluation.csv", index=False, encoding="utf-8-sig")
