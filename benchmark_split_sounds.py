import os
import librosa
import torch
import numpy as np
import joblib
import warnings

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

DATA_DIR = "dataset_augmented"
CACHE_FILE = "features_layers_cache.pkl"
LAYERS_TO_TEST = [8, 12, 16, 20] # Тестуємо різні глибини нейромережі

# ==========================================
# 1. ЕКСТРАКЦІЯ ШАРІВ (З КЕШУВАННЯМ)
# ==========================================
if os.path.exists(CACHE_FILE):
    print(f"📁 Завантаження кешу з ознаками різних шарів з '{CACHE_FILE}'...")
    data_dict = joblib.load(CACHE_FILE)
else:
    print("⏳ Кеш не знайдено. Екстракція ознак для різних шарів (це займе 10-15 хвилин)...")
    processor = Wav2Vec2Processor.from_pretrained("anton-l/wav2vec2-large-xlsr-53-ukrainian")
    wav2vec_model = Wav2Vec2Model.from_pretrained("anton-l/wav2vec2-large-xlsr-53-ukrainian")
    
    # Зберігатимемо дані у словник: { 'singles': { 'layer_8': [], ... }, 'pairs': { ... } }
    data_dict = {
        'singles': {'X': {l: [] for l in LAYERS_TO_TEST}, 'y': [], 'groups': []},
        'pairs':   {'X': {l: [] for l in LAYERS_TO_TEST}, 'y': [], 'groups': []}
    }
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]
    for i, filename in enumerate(files):
        parts = filename.split("_")
        if len(parts) >= 3:
            speaker_id = parts[0].lower()
            label = parts[1].lower()
            file_path = os.path.join(DATA_DIR, filename)
            
            # Визначаємо категорію: 1 літера = 'singles', більше = 'pairs'
            category = 'singles' if len(label) == 1 else 'pairs'
            
            try:
                speech, _ = librosa.load(file_path, sr=16000)
                speech, _ = librosa.effects.trim(speech, top_db=20)
                speech = librosa.util.normalize(speech)
                
                inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = wav2vec_model(**inputs, output_hidden_states=True)
                
                # Витягуємо дані для кожного шару з нашого списку
                for layer in LAYERS_TO_TEST:
                    features = outputs.hidden_states[layer].squeeze(0)
                    
                    if category == 'pairs':
                        # Для складів робимо Temporal Pooling (розбиття на 3 частини)
                        if features.shape[0] < 3:
                            features = features.repeat(3, 1)
                        chunks = torch.tensor_split(features, 3, dim=0)
                        emb = torch.cat([chunk.mean(dim=0) for chunk in chunks]).numpy()
                    else:
                        # Для одиничних звуків звичайного середнього достатньо
                        emb = features.mean(dim=0).numpy()
                        
                    data_dict[category]['X'][layer].append(emb)
                
                # Додаємо мітки лише один раз на файл
                data_dict[category]['y'].append(label)
                data_dict[category]['groups'].append(speaker_id)
                
            except Exception as e:
                pass
                
        if (i + 1) % 100 == 0:
            print(f"Оброблено {i + 1} / {len(files)} файлів...")
            
    # Конвертуємо списки в numpy arrays
    for cat in ['singles', 'pairs']:
        data_dict[cat]['y'] = np.array(data_dict[cat]['y'])
        data_dict[cat]['groups'] = np.array(data_dict[cat]['groups'])
        for l in LAYERS_TO_TEST:
            data_dict[cat]['X'][l] = np.array(data_dict[cat]['X'][l])
            
    joblib.dump(data_dict, CACHE_FILE)
    print(f"✅ Усі ознаки збережено у '{CACHE_FILE}'.")

# ==========================================
# 2. ОЦІНКА МЕТОДІВ (БЕЗ ВИТОКУ ДАНИХ)
# ==========================================
print("\n" + "="*60)
print(" РОЗПОЧИНАЄМО БЕНЧМАРКІНГ (SPEAKER-INDEPENDENT)".center(60))
print("="*60)

# Тестуємо дві найстабільніші архітектури
models_to_test = {
    "SVM (RBF)": SVC(kernel='rbf', C=10.0, class_weight='balanced', random_state=42),
    "MLP (NeuNet)": MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=42)
}

gkf = GroupKFold(n_splits=5) # 5 незалежних перевірок на різних дітях

for category in ['singles', 'pairs']:
    print(f"\n🔹 АНАЛІЗ КАТЕГОРІЇ: {category.upper()} (Файлів: {len(data_dict[category]['y'])})")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(data_dict[category]['y'])
    groups = data_dict[category]['groups']
    
    best_acc = 0
    best_combo = ""
    
    for layer in LAYERS_TO_TEST:
        X = data_dict[category]['X'][layer]
        
        for model_name, clf in models_to_test.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', clf)
            ])
            
            # Оцінюємо модель (5 фолдів, витоку даних немає)
            scores = cross_val_score(pipeline, X, y_encoded, groups=groups, cv=gkf, n_jobs=-1)
            mean_acc = scores.mean() * 100
            
            print(f"   Шари {layer:2} + {model_name:12} -> Точність: {mean_acc:.2f}%")
            
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_combo = f"Шар {layer} + {model_name}"
                
    print(f"🌟 НАЙКРАЩЕ ДЛЯ {category.upper()}: {best_combo} ({best_acc:.2f}%)")
print("="*60)