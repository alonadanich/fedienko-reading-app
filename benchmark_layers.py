import os
import librosa
import torch
import numpy as np
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings

# Вимикаємо попередження для чистоти консолі
warnings.filterwarnings("ignore")

print("Завантаження Wav2Vec 2.0 (XLSR-53 Ukrainian)...")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
processor = Wav2Vec2Processor.from_pretrained("anton-l/wav2vec2-large-xlsr-53-ukrainian")
model = Wav2Vec2Model.from_pretrained("anton-l/wav2vec2-large-xlsr-53-ukrainian")

# Вкажіть шлях до вашої папки з аудіо (де лежать папки з класами "a", "u", "au" тощо)
DATASET_PATH = "dataset_augmented" 

def extract_all_strategies(audio_path):
    # 1. Завантаження та препроцесинг
    speech, rate = librosa.load(audio_path, sr=16000)
    speech_trimmed, _ = librosa.effects.trim(speech, top_db=20)
    if len(speech_trimmed) > 1000:
        speech = speech_trimmed
    speech = librosa.util.normalize(speech)
    
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    
    # 2. Магія: просимо модель видати ВСІ приховані шари
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # outputs.hidden_states - це кортеж із 25 тензорів: 
    # [0] - CNN вихід, [1] - шар 1 ... [24] - шар 24
    hs = outputs.hidden_states

    # Допоміжні функції для пулінгу
    def get_mean(layer_idx):
        return hs[layer_idx].mean(dim=1).squeeze().numpy()
        
    def get_std(layer_idx):
        return hs[layer_idx].std(dim=1).squeeze().numpy()

    # 3. Формуємо вектори для всіх 6 стратегій ментора
    features = {
        "CNN + Mean": get_mean(0),
        "Layer 12 + Mean": get_mean(12),
        "Layer 18 + Mean": get_mean(18),
        "Layer 24 + Mean": get_mean(24),
        "Layer 24 + Mean+Std": np.concatenate([get_mean(24), get_std(24)]),
        "Layers 12+18+24 + Mean": np.concatenate([get_mean(12), get_mean(18), get_mean(24)])
    }
    
    return features

# ==========================================
# ОСНОВНИЙ ЦИКЛ ЗБОРУ ДАНИХ
# ==========================================
data_store = {
    "CNN + Mean": [], "Layer 12 + Mean": [], "Layer 18 + Mean": [],
    "Layer 24 + Mean": [], "Layer 24 + Mean+Std": [], "Layers 12+18+24 + Mean": []
}
labels = []

print(f"Починаємо екстракцію ознак з папки {DATASET_PATH}...")
# Збираємо всі файли (для пришвидшення можна обмежити кількість файлів на клас для тесту)
valid_extensions = ('.wav', '.m4a')

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith(valid_extensions):
            file_path = os.path.join(root, file)
            try:
                label = file.split('_')[1].lower() # Витягуємо мітку з назви файлу
            except IndexError:
                continue
            
            try:
                feats = extract_all_strategies(file_path)
                for strategy in data_store.keys():
                    data_store[strategy].append(feats[strategy])
                labels.append(label)
            except Exception as e:
                print(f"Помилка обробки {file}: {e}")

# Кодуємо мітки (labels) у числа
le = LabelEncoder()
y = le.fit_transform(labels)

# ==========================================
# ТРЕНУВАННЯ ТА ОЦІНКА SVM
# ==========================================
results = []

print("\n--- РЕЗУЛЬТАТИ БЕНЧМАРКІНГУ АРХІТЕКТУР ---")
for strategy, X_list in data_store.items():
    X = np.array(X_list)
    
    # Розбиваємо дані на тренувальну та тестову вибірки (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Тренуємо SVM з лінійним ядром
    clf = SVC(kernel='linear', C=10.0, probability=True)
    clf.fit(X_train, y_train)
    
    # Оцінюємо точність
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    
    vector_size = X.shape[1]
    results.append({"Стратегія екстракції": strategy, "Розмірність вектора": vector_size, "Точність SVM (%)": round(acc, 2)})
    print(f"{strategy:<25} | Розмір вектора: {vector_size:<4} | Точність: {acc:.2f}%")

# Зберігаємо гарну табличку для диплому у CSV
df_results = pd.DataFrame(results)
df_results.to_csv("wav2vec_layers_benchmark.csv", index=False)
print("\n✅ Таблицю збережено у файл 'wav2vec_layers_benchmark.csv'. Використайте ці дані для Розділу 4!")