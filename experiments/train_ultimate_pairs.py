import os
import librosa
import torch
import numpy as np
import joblib
import warnings

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

DATA_DIR = "dataset_augmented"
CACHE_FILE = "ultimate_pairs_cache.pkl"

print("1. Завантаження Wav2Vec 2.0...")
processor = Wav2Vec2Processor.from_pretrained("anton-l/wav2vec2-large-xlsr-53-ukrainian")
wav2vec_model = Wav2Vec2Model.from_pretrained("anton-l/wav2vec2-large-xlsr-53-ukrainian")

def get_ultimate_features(file_path):
    """Multi-Layer Fusion + Statistical Temporal Pooling (Mean & Std)"""
    speech, _ = librosa.load(file_path, sr=16000)
    speech, _ = librosa.effects.trim(speech, top_db=20)
    speech = librosa.util.normalize(speech)
    
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wav2vec_model(**inputs, output_hidden_states=True)
    
    # Витягуємо одразу два шари: 12-й та 16-й
    layer_12 = outputs.hidden_states[12].squeeze(0)
    layer_16 = outputs.hidden_states[16].squeeze(0)
    
    if layer_12.shape[0] < 3:
        layer_12 = layer_12.repeat(3, 1)
        layer_16 = layer_16.repeat(3, 1)
        
    chunks_12 = torch.tensor_split(layer_12, 3, dim=0)
    chunks_16 = torch.tensor_split(layer_16, 3, dim=0)
    
    # Для кожної з 3 частин беремо і СЕРЕДНЄ, і ВІДХИЛЕННЯ (std)
    stats = []
    for c12, c16 in zip(chunks_12, chunks_16):
        stats.append(c12.mean(dim=0))
        stats.append(c12.std(dim=0, unbiased=False))
        stats.append(c16.mean(dim=0))
        stats.append(c16.std(dim=0, unbiased=False))
        
    combined_emb = torch.cat(stats)
    return combined_emb.numpy()

if os.path.exists(CACHE_FILE):
    print("📁 Знайдено кеш ультимативних ознак. Завантажуємо...")
    X, y, groups = joblib.load(CACHE_FILE)
else:
    print("⏳ Кешу немає. Екстракція надскладних векторів для складів (це займе 10-15 хвилин)...")
    X, y, groups = [], [], []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]
    
    for i, filename in enumerate(files):
        parts = filename.split("_")
        if len(parts) >= 3:
            label = parts[1].lower()
            if len(label) < 2:  # Пропускаємо одиничні звуки
                continue
                
            speaker_id = parts[0].lower()
            file_path = os.path.join(DATA_DIR, filename)
            
            try:
                emb = get_ultimate_features(file_path)
                X.append(emb)
                y.append(label)
                groups.append(speaker_id)
            except Exception as e:
                pass
                
        if (i + 1) % 100 == 0:
            print(f"Оброблено {i + 1} / {len(files)} файлів...")
            
    X, groups = np.array(X), np.array(groups)
    joblib.dump((X, y, groups), CACHE_FILE)
    print("✅ Ультимативний кеш збережено.")

print("\n2. Підготовка до глибокого тюнінгу...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"📊 Дані: {X.shape[0]} складів. Розмірність вектора: {X.shape[1]} ознак! (Гігантський вектор)")

# Оскільки вектор тепер має понад 12000 ознак, ми обов'язково додаємо PCA для стиснення
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(random_state=42)),
    ('mlp', MLPClassifier(solver='adam', early_stopping=True, random_state=42))
])

param_grid = {
    'pca__n_components': [512, 1024], # Стискаємо акустичний шум
    'mlp__hidden_layer_sizes': [(512, 256, 128), (1024, 512, 256)], 
    'mlp__alpha': [0.1, 1.0],         # Жорстка регуляризація
    'mlp__activation': ['tanh']
}

gkf = GroupKFold(n_splits=5)
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=gkf, scoring='accuracy', n_jobs=-1, verbose=1)

print("\n🚀 Запуск автоматичного пошуку найкращої моделі...")
grid_search.fit(X, y_encoded, groups=groups)

print("\n" + "="*60)
print("🏆 НАЙКРАЩА УЛЬТИМАТИВНА АРХІТЕКТУРА ДЛЯ СКЛАДІВ:")
print(grid_search.best_params_)
print(f"📈 СЕРЕДНЯ ТОЧНІСТЬ З НОВИМ ВЕКТОРОМ: {grid_search.best_score_ * 100:.2f}%")
print("="*60)