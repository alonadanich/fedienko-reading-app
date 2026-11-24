import os
import joblib
import numpy as np
import warnings
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Використовуємо ImbPipeline замість звичайного sklearn Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

CACHE_FILE = "features_layers_cache.pkl"

print("1. Завантаження даних з кешу...")
if not os.path.exists(CACHE_FILE):
    print(f"Помилка: Кеш {CACHE_FILE} не знайдено.")
    exit()

data_dict = joblib.load(CACHE_FILE)

# Використовуємо наш найкращий 12-й шар для складів (pairs)
X = data_dict['pairs']['X'][12]
y = data_dict['pairs']['y']
groups = data_dict['pairs']['groups']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"📊 Дані: {X.shape[0]} зразків, {X.shape[1]} ознак (Шар 12 + Temporal Pooling).")

# Налаштовуємо SMOTE на меншу кількість сусідів (k_neighbors=2) 
# для безпеки, якщо деякі склади вимовлялися дуже мало разів
smote = SMOTE(k_neighbors=2, random_state=42)
smote_tomek = SMOTETomek(smote=smote, random_state=42)

# Створюємо супер-конвеєр
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    # Стискаємо 3072 ознаки до 256, щоб CatBoost та SMOTE не перенавчилися на шумі 
    # і працювали швидко
    ('pca', PCA(n_components=256, random_state=42)), 
    ('smotetomek', smote_tomek),
    ('catboost', CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        task_type='CPU', # Замініть на 'GPU', якщо налаштована відеокарта
        thread_count=-1,
        verbose=False,
        random_state=42
    ))
])

gkf = GroupKFold(n_splits=5)
scores = []

print("\n🚀 Запуск 5-Fold Cross-Validation (CatBoost + SMOTETomek)...")
print("Це може зайняти кілька хвилин. Чекаємо на результати кожного фолду:")

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_encoded, groups), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    scores.append(acc)
    print(f"   Фолд {fold} -> Точність: {acc * 100:.2f}%")

final_accuracy = np.mean(scores)

print("\n" + "="*60)
print(f"🏆 ФІНАЛЬНА ТОЧНІСТЬ CATBOOST (Speaker-Independent): {final_accuracy * 100:.2f}%")
print("="*60)

# Збереження моделі для подальшого використання у застосунку
joblib.dump(pipeline, "pipeline_catboost_pairs.pkl")
joblib.dump(le, "le_catboost_pairs.pkl")
print("✅ Модель CatBoost успішно збережена!")