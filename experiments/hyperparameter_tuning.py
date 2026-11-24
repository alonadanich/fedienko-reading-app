import os
import librosa
import torch
import numpy as np
import joblib
import warnings

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

DATA_DIR = "dataset_augmented"
CACHE_FILE = "features_cache.pkl"

# ==========================================
# 1. ЕКСТРАКЦІЯ АБО ЗАВАНТАЖЕННЯ ОЗНАК (КЕШУВАННЯ)
# ==========================================
if os.path.exists(CACHE_FILE):
    print(f"📁 Знайдено кеш '{CACHE_FILE}'. Завантаження ознак (це займе секунду)...")
    X, y, groups = joblib.load(CACHE_FILE)
else:
    print("⏳ Кеш не знайдено. Починаємо екстракцію ознак з 12-го шару (це займе 10-15 хвилин)...")
    processor = Wav2Vec2Processor.from_pretrained("anton-l/wav2vec2-large-xlsr-53-ukrainian")
    wav2vec_model = Wav2Vec2Model.from_pretrained("anton-l/wav2vec2-large-xlsr-53-ukrainian")
    
    X, y, groups = [], [], []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]
    
    for i, filename in enumerate(files):
        parts = filename.split("_")
        if len(parts) >= 3:
            speaker_id = parts[0].lower()
            label = parts[1].lower()
            file_path = os.path.join(DATA_DIR, filename)
            
            try:
                speech, _ = librosa.load(file_path, sr=16000)
                speech, _ = librosa.effects.trim(speech, top_db=20)
                speech = librosa.util.normalize(speech)
                
                inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = wav2vec_model(**inputs, output_hidden_states=True)
                
                emb = outputs.hidden_states[12].mean(dim=1).squeeze().numpy()
                X.append(emb)
                y.append(label)
                groups.append(speaker_id)
            except Exception as e:
                pass
                
        if (i + 1) % 100 == 0:
            print(f"Оброблено {i + 1} / {len(files)} файлів...")
            
    X, groups = np.array(X), np.array(groups)
    joblib.dump((X, y, groups), CACHE_FILE)
    print(f"✅ Ознаки успішно збережено у '{CACHE_FILE}'.")

# ==========================================
# 2. ПІДГОТОВКА ДО ПОШУКУ
# ==========================================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\n📊 Всього даних: {len(X)} зразків. Унікальних дикторів: {len(np.unique(groups))}")

# Створюємо Конвеєр (Pipeline): Масштабування -> Зменшення розмірності -> Класифікатор
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(random_state=42)),
    ('svm', SVC(class_weight='balanced', probability=True, random_state=42))
])

# Прописуємо гіперпараметри, які хочемо протестувати (Регуляризація)
param_grid = {
    'pca__n_components': [128, 256, 512],  # Стискаємо вектор, щоб видалити "шум диктора"
    'svm__kernel': ['linear', 'rbf'],      # Тестуємо лінійні та нелінійні межі
    'svm__C': [0.01, 0.1, 1.0, 10.0]       # Сила регуляризації (менше = сильніше ігнорує викиди)
}

# Використовуємо GroupKFold на 5 згинів (folds)
# Це означає, що модель 5 разів перевірить себе на 5 різних наборах ізольованих дітей!
gkf = GroupKFold(n_splits=5)

# Налаштовуємо "Шукач"
grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=gkf,
    scoring='accuracy',
    n_jobs=-1,  # Використовувати всі ядра процесора для швидкості
    verbose=2   # Виводити процес на екран
)

# ==========================================
# 3. СТАРТ ЕКСПЕРИМЕНТУ
# ==========================================
print("\n🚀 Запуск автоматичного пошуку гіперпараметрів (GridSearch) без витоку даних...")
grid_search.fit(X, y_encoded, groups=groups)

print("\n" + "="*50)
print("🏆 НАЙКРАЩІ ЗНАЙДЕНІ ПАРАМЕТРИ:")
print(grid_search.best_params_)
print(f"📈 НАЙКРАЩА СЕРЕДНЯ ТОЧНІСТЬ (Speaker-Independent): {grid_search.best_score_ * 100:.2f}%")
print("="*50)

# Зберігаємо найкращу знайдену модель (весь конвеєр разом зі Scaler і PCA)
best_model = grid_search.best_estimator_
joblib.dump(best_model, "fedienko_best_pipeline.pkl")
joblib.dump(le, "label_encoder_pipeline.pkl")
print("\n✅ Найкращу модель успішно збережено як 'fedienko_best_pipeline.pkl'.")