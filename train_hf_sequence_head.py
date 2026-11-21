import os
import librosa
import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
MODEL_ID = "anton-l/wav2vec2-large-xlsr-53-ukrainian"
DATA_DIR = "dataset_augmented"
OUT_DIR = os.path.join("external_models", "hf_wav2vec2_seq_clf")

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

audios, labels, speakers = [], [], []
for fname in os.listdir(DATA_DIR):
    if not fname.endswith(".wav"):
        continue
    parts = fname.split("_")
    if len(parts) >= 3:
        speech, _ = librosa.load(os.path.join(DATA_DIR, fname), sr=16000)
        speech, _ = librosa.effects.trim(speech, top_db=20)
        audios.append(librosa.util.normalize(speech))
        speakers.append(parts[0].lower())
        labels.append(parts[1].lower())

le = LabelEncoder()
y = le.fit_transform(labels)
id2label = {int(i): lbl for i, lbl in enumerate(le.classes_)}
label2id = {lbl: int(i) for i, lbl in enumerate(le.classes_)}

# Чесне Speaker-Independent розбиття по дикторах
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(audios, y, groups=speakers))

train_ds = Dataset.from_dict({"audio": [audios[i] for i in train_idx], "labels": y[train_idx].tolist()})
test_ds = Dataset.from_dict({"audio": [audios[i] for i in test_idx], "labels": y[test_idx].tolist()})


def preprocess(batch):
    enc = processor(
        batch["audio"], sampling_rate=16000, padding="max_length", max_length=32000, truncation=True
    )
    enc["labels"] = batch["labels"]
    return enc


train_ds = train_ds.map(preprocess, batched=True, batch_size=32, remove_columns=["audio"])
test_ds = test_ds.map(preprocess, batched=True, batch_size=32, remove_columns=["audio"])

# Використовуємо штатне зважування всіх шарів із документації Wav2Vec2Config!
config = Wav2Vec2Config.from_pretrained(
    MODEL_ID,
    num_labels=len(le.classes_),
    id2label=id2label,
    label2id=label2id,
    use_weighted_layer_sum=True,
    classifier_proj_size=256,
)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID, config=config)

# КЛЮЧОВА ВІДМІННІСТЬ: заморожуємо всі 300М ваг трансформера, навчаємо лише штатну голову HF і ваги шарів!
model.freeze_base_model()


def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    return {"accuracy": float((preds == eval_pred.label_ids).mean())}


args = TrainingArguments(
    output_dir="./hf_head_checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=8,
    load_best_model_at_end=True,
    report_to="none",
)

trainer = Trainer(
    model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds, compute_metrics=compute_metrics
)
trainer.train()
res = trainer.evaluate()
print(f"Чесна Speaker-Independent точність штатної голови HF: {res['eval_accuracy'] * 100:.2f}%")

os.makedirs(OUT_DIR, exist_ok=True)
trainer.save_model(OUT_DIR)
processor.save_pretrained(OUT_DIR)
print(f"Модель збережено для підключення через UI: {OUT_DIR}")