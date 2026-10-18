import os
import librosa
import torch
import pandas as pd
from transformers import WhisperProcessor, WhisperForConditionalGeneration

print("Завантаження моделі... Це може зайняти хвилину.")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
print("Модель завантажено успішно!\n")

# Жорстко фіксуємо українську мову для процесора
forced_decoder_ids = processor.get_decoder_prompt_ids(language="uk", task="transcribe")

folder_path = "test_audio"
results = []

for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)

        try:
            audio_array, sampling_rate = librosa.load(file_path, sr=16000)
            input_features = processor(
                audio_array, sampling_rate=sampling_rate, return_tensors="pt"
            ).input_features

            # ЗМІНА ТУТ: Примусово передаємо українські токени прямо у функцію генерації
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=255,
            )

            transcription = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            print(f"Файл: {filename} | Розпізнано ШІ: {transcription}")

            results.append(
                {"Назва файлу": filename, "Розпізнаний текст": transcription}
            )

        except Exception as e:
            print(f"Помилка при обробці файлу {filename}: {e}")

df = pd.DataFrame(results)
output_csv = "whisper_results.csv"
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"\nГотово! Результати збережено у файл: {output_csv}")
