import os
import librosa
import torch
import pandas as pd
import re  # ДОДАНО: бібліотека для роботи з регулярними виразами (потрібна для очищення тексту)
from transformers import WhisperProcessor, WhisperForConditionalGeneration


# 1. ФУНКЦІЯ ОЧИЩЕННЯ ТЕКСТУ (Нормалізація)
# Цей фрагмент коду ви зможете додати в ДОДАТОК Б вашої дипломної роботи
def normalize_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()  # Переводимо всі літери в нижній регістр
    text = re.sub(r"[^\w\s]", "", text)  # Видаляємо знаки пунктуації (?, !, ...)
    text = re.sub(
        r"([а-яіїєґ])\1+", r"\1", text
    )  # Видаляємо дублювання літер (ааа -> а)
    return text.strip()


# 2. ЗАВАНТАЖЕННЯ МОДЕЛІ
print("Завантаження моделі...")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="uk", task="transcribe")
print("Модель завантажено успішно!\n")

folder_path = "test_audio"
results = []

# 3. ЦИКЛ ОБРОБКИ ФАЙЛІВ
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)

        try:
            audio_array, sampling_rate = librosa.load(file_path, sr=16000)
            input_features = processor(
                audio_array, sampling_rate=sampling_rate, return_tensors="pt"
            ).input_features

            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=255,
            )

            # Оригінальний текст від ШІ
            transcription = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            # ДОДАНО: Застосовуємо функцію очищення до тексту
            cleaned_transcription = normalize_text(transcription)

            print(
                f"Файл: {filename} | Оригінал: {transcription} | Очищено: {cleaned_transcription}"
            )

            # Записуємо в таблицю відразу три колонки
            results.append(
                {
                    "Назва файлу": filename,
                    "Розпізнаний текст": transcription,
                    "Очищений текст": cleaned_transcription,
                }
            )

        except Exception as e:
            print(f"Помилка при обробці файлу {filename}: {e}")

# 4. ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ
df = pd.DataFrame(results)
output_csv = "whisper_results_cleaned.csv"
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"\nГотово! Результати збережено у файл: {output_csv}")
