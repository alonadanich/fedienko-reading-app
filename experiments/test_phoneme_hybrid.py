import os
import librosa
import torch

# Вказуємо Python, де лежить бібліотека eSpeak (після її встановлення)
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import warnings
warnings.filterwarnings("ignore")

print("Завантаження спеціалізованої фонемної моделі...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

def recognize_phonemes(file_path):
    # Завантажуємо аудіо
    speech, rate = librosa.load(file_path, sr=16000)
    speech_trimmed, _ = librosa.effects.trim(speech, top_db=20)
    speech = librosa.util.normalize(speech_trimmed)
    
    # Пропускаємо через фонемну модель
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Декодуємо результат у фонеми (IPA)
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# ==========================================
# ТЕСТУВАННЯ
# ==========================================
DATA_DIR = "dataset_augmented"
test_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")][:10] # Беремо перші 10 файлів для тесту

print("\n--- РЕЗУЛЬТАТИ РОЗПІЗНАВАННЯ ФОНЕМ ---")
for file in test_files:
    file_path = os.path.join(DATA_DIR, file)
    label = file.split('_')[1] # Очікуваний звук
    
    try:
        phonemes = recognize_phonemes(file_path)
        print(f"Файл: {file:25} | Очікували: {label:5} | Модель почула фонеми: [{phonemes}]")
    except Exception as e:
        pass