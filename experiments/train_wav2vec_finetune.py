import os
import librosa
import torch
import numpy as np
import pandas as pd
from datasets import Dataset, ClassLabel
from sklearn.preprocessing import LabelEncoder
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import warnings

warnings.filterwarnings("ignore")

print("Підготовка до Fine-Tuning Wav2Vec 2.0...")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
MODEL_ID = "anton-l/wav2vec2-large-xlsr-53-ukrainian"
DATASET_PATH = "dataset_augmented"

# 1. Завантажуємо процесор
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

# 2. Збираємо дані
print("Зчитування аудіофайлів...")
audio_arrays = []
labels = []
valid_extensions = (".wav", ".m4a")

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith(valid_extensions):
            file_path = os.path.join(root, file)
            # Мітка береться з назви файлу (як ми робили раніше)
            try:
                label = file.split("_")[1].lower()
            except IndexError:
                continue  # Пропускаємо файли з неправильними назвами

            try:
                speech, rate = librosa.load(file_path, sr=16000)
                speech_trimmed, _ = librosa.effects.trim(speech, top_db=20)
                if len(speech_trimmed) > 1000:
                    speech = speech_trimmed
                speech = librosa.util.normalize(speech)

                audio_arrays.append(speech)
                labels.append(label)
            except Exception as e:
                pass

print(f"Успішно завантажено {len(audio_arrays)} файлів.")

# 3. Кодуємо мітки
le = LabelEncoder()
y = le.fit_transform(labels)
num_classes = len(le.classes_)
print(f"Кількість унікальних класів: {num_classes}")

# Створюємо словники для моделі (id -> label, label -> id)
id2label = {i: label for i, label in enumerate(le.classes_)}
label2id = {label: i for i, label in enumerate(le.classes_)}

# 4. Створюємо Dataset у форматі HuggingFace
data_dict = {"audio": audio_arrays, "label": y}
hf_dataset = Dataset.from_dict(data_dict)

# ВИПРАВЛЕННЯ: Конвертуємо колонку 'label' у спеціальний тип ClassLabel
hf_dataset = hf_dataset.cast_column(
    "label", ClassLabel(num_classes=num_classes, names=list(le.classes_))
)

# Тепер розбиваємо на train / test (stratify спрацює без помилок)
hf_dataset = hf_dataset.train_test_split(
    test_size=0.2, seed=42, stratify_by_column="label"
)


# 5. Функція для перетворення аудіо у формат для нейромережі
def preprocess_function(examples):
    inputs = processor(
        examples["audio"],
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",  # Додаємо паддінг, щоб всі вектори були однакової довжини
        max_length=48000,  # Максимальна довжина аудіо (3 секунди = 16000 * 3)
        truncation=True,
    )
    inputs["labels"] = examples["label"]
    return inputs


print("Токенізація датасету...")
encoded_dataset = hf_dataset.map(
    preprocess_function, batched=True, remove_columns=["audio", "label"]
)

# 6. Завантажуємо модель з Classification Head
print("Завантаження моделі Wav2Vec2ForSequenceClassification...")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=num_classes, id2label=id2label, label2id=label2id
)

# Заморожуємо базові шари (Feature Extractor), щоб не зламати попередньо натреновану акустику
model.freeze_feature_encoder()


# 7. Метрики для оцінки
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    acc = (predictions == eval_pred.label_ids).mean()
    return {"accuracy": acc}


# 8. Налаштування тренування
training_args = TrainingArguments(
    output_dir="./wav2vec2-finetuned-fedienko",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=1,  # <--- Зменшили з 8 до 1 (щоб не переповнювати пам'ять)
    per_device_eval_batch_size=1,   # <--- Зменшили з 8 до 1
    gradient_accumulation_steps=8,  # <--- Накопичуємо градієнти, щоб модель думала, що батч все ще 8
    num_train_epochs=3, 
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_steps=10,
    report_to="none",               # <--- Вимикаємо сторонні логери, які теж можуть мовчки крашити скрипт на Windows
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    compute_metrics=compute_metrics,
)

# 9. Запускаємо Fine-Tuning
print("Починаємо тренування (Fine-Tuning)! Це може зайняти певний час...")
trainer.train()

# Оцінюємо фінальну модель
print("Фінальна оцінка:")
eval_results = trainer.evaluate()
print(eval_results)

# Зберігаємо модель
print("Збереження моделі...")
trainer.save_model("./best_wav2vec_model")
processor.save_pretrained("./best_wav2vec_model")
print("✅ Fine-Tuning завершено. Модель збережена у папку 'best_wav2vec_model'.")
