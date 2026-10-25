import os
import librosa
import torch
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from scipy.spatial.distance import cosine

# Вимикаємо попередження
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

print("Завантаження моделі Wav2Vec 2.0...")
model_name = "anton-l/wav2vec2-large-xlsr-53-ukrainian"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)
print("Модель готова. Починаємо аналіз датасету...")


def get_embedding(file_path):
    audio_array, sr = librosa.load(file_path, sr=16000)
    audio_array, _ = librosa.effects.trim(audio_array, top_db=20)
    inputs = processor(
        audio_array, sampling_rate=16000, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


# Шляхи до папок
mother_dir = "dataset/mother"
child_dir = "dataset/child"

# Отримуємо список усіх звуків (по назвах файлів без розширення)
# Наприклад: ['a', 'o', 'ma', 'ba' ...]
sounds = [f.split(".")[0] for f in os.listdir(mother_dir) if f.endswith(".wav")]

results = []

for target in sounds:
    mother_file = os.path.join(mother_dir, f"{target}.wav")
    child_correct_file = os.path.join(child_dir, f"{target}.wav")

    # 1. Рахуємо правильну вимову (True Positive)
    emb_mother = get_embedding(mother_file)
    emb_child_correct = get_embedding(child_correct_file)

    true_positive_score = max(0.0, (1 - cosine(emb_mother, emb_child_correct)) * 100.0)

    # 2. Рахуємо помилкову вимову (False Positives) - порівнюємо еталон з УСІМА ІНШИМИ записами дитини
    highest_false_positive = 0.0
    worst_confused_sound = ""

    for other_sound in sounds:
        if other_sound == target:
            continue

        child_wrong_file = os.path.join(child_dir, f"{other_sound}.wav")
        if not os.path.exists(child_wrong_file):
            continue

        emb_child_wrong = get_embedding(child_wrong_file)
        score = max(0.0, (1 - cosine(emb_mother, emb_child_wrong)) * 100.0)

        if score > highest_false_positive:
            highest_false_positive = score
            worst_confused_sound = other_sound

    # 3. Вираховуємо ідеальний поріг (рівно посередині між правильною і найгіршою помилковою)
    suggested_threshold = (true_positive_score + highest_false_positive) / 2

    # Якщо правильна вимова отримала нижчий бал, ніж помилкова (аномалія)
    status = "OK" if true_positive_score > highest_false_positive else "WARNING"

    results.append(
        {
            "Звук (Target)": target,
            "Правильно сказано (TP)": round(true_positive_score, 1),
            "Хибний максимум (FP)": round(highest_false_positive, 1),
            "З чим сплутано": worst_confused_sound,
            "Рекомендований Поріг": round(suggested_threshold, 1),
            "Статус": status,
        }
    )
    print(f"Оброблено: {target} | Поріг: {round(suggested_threshold, 1)}")

# Зберігаємо результати у файл
df = pd.DataFrame(results)
df.to_csv("thresholds_report.csv", index=False, encoding="utf-8-sig")
print("\nАналіз завершено! Результати збережено у 'thresholds_report.csv'")
