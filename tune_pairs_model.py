import os
import joblib
import numpy as np
import warnings
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")

CACHE_FILE = "features_layers_cache.pkl"

print("1. Завантаження даних з кешу...")
if not os.path.exists(CACHE_FILE):
    print(f"Помилка: Кеш {CACHE_FILE} не знайдено.")
    exit()

data_dict = joblib.load(CACHE_FILE)

# Беремо виключно ПАРНІ звуки (pairs) і наш найкращий 12-й шар
X = data_dict['pairs']['X'][12]
y = data_dict['pairs']['y']
groups = data_dict['pairs']['groups']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"📊 Дані для складів завантажено: {X.shape[0]} зразків, {X.shape[1]} ознак.")
print(f"Унікальних дикторів: {len(np.unique(groups))}")

# 2. Налаштовуємо пошук гіперпараметрів
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        max_iter=400, 
        solver='adam', 
        early_stopping=True, # Запобігає перенавчанню
        random_state=42
    ))
])

param_grid = {
    # Архітектура нейромережі (ширина і глибина)
    'mlp__hidden_layer_sizes': [
        (256,),             # Базовий варіант
        (512, 256),         # Глибша мережа
        (256, 128),         # Звуження для видалення шуму
        (512, 256, 128)     # Дуже глибока
    ],
    # Сила L2-регуляризації (більше значення = сильніше ігнорує тембр диктора)
    'mlp__alpha': [0.001, 0.01, 0.1, 1.0],
    # Функція активації
    'mlp__activation': ['relu', 'tanh']
}

gkf = GroupKFold(n_splits=5)

print("\n🚀 Запуск глибокого тюнінгу для складів (Speaker-Independent)...")
print("Буде перевірено 32 різні архітектури. Зачекайте кілька хвилин...")

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=gkf,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y_encoded, groups=groups)

print("\n" + "="*60)
print("🏆 НАЙКРАЩА АРХІТЕКТУРА ДЛЯ СКЛАДІВ:")
print(grid_search.best_params_)
print(f"📈 НОВА СЕРЕДНЯ ТОЧНІСТЬ (Speaker-Independent): {grid_search.best_score_ * 100:.2f}%")
print("="*60)

# Зберігаємо переможця
best_model = grid_search.best_estimator_.named_steps['mlp']
best_scaler = grid_search.best_estimator_.named_steps['scaler']

joblib.dump(best_model, "model_mlp_pairs.pkl")
joblib.dump(best_scaler, "scaler_pairs.pkl")
joblib.dump(le, "le_pairs.pkl")

print("\n✅ Найкращу модель для парних звуків оновлено та збережено!")
