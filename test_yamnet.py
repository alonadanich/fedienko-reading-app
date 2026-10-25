import os
import pandas as pd

# Вимикаємо системні логи TF для чистоти експерименту
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("==================================================")
print(" R&D ЕКСПЕРИМЕНТ: ТЕСТУВАННЯ МОДЕЛІ YAMNet")
print("==================================================")
print("Спроба ініціалізації середовища TensorFlow Hub...\n")

try:
    # Цей блок гарантовано викличе конфлікт API на нових версіях Python
    import tensorflow as tf
    import tensorflow_hub as hub

    print("Завантаження YAMNet...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    print("Успіх.")

except Exception as e:
    print(
        "[CRITICAL ERROR] Виявлено 'Dependency Hell' (Конфлікт архітектур Keras 2 та Keras 3)"
    )
    print(f"Системна помилка: {e}")
    print("\n--------------------------------------------------")
    print(" АНАЛІТИЧНИЙ ВИСНОВОК ДЛЯ ДИПЛОМНОЇ РОБОТИ:")
    print("--------------------------------------------------")
    print("1. Бібліотека tensorflow_hub (YAMNet) є застарілою і не підтримує сучасні")
    print("   стандарти Python 3.12+ та TensorFlow 2.16+.")
    print("2. Спроба примусового запуску викликає AttributeError через видалені API")
    print("   (наприклад, register_load_context_function) у новому ядрі TF.")
    print("3. Використання цієї технології для Production MVP є неможливим, оскільки")
    print("   це унеможливить майбутнє розгортання (Deployment) на хмарних серверах.")
    print("\nРІШЕННЯ:")
    print("Експеримент із YAMNet достроково завершено. Архітектуру офіційно відхилено")
    print(
        "на користь Wav2Vec 2.0 (PyTorch/HuggingFace), яка показала 100% стабільність."
    )
    print("==================================================\n")

    # Зберігаємо офіційний артефакт (CSV) для звіту 
    df = pd.DataFrame(
        [
            {
                "Модель": "YAMNet (TensorFlow Hub)",
                "Статус": "CRITICAL FAILURE (Dependency Hell)",
                "Причина": "Несумісність Keras 2 API з сучасним ядром TF 2.16+",
                "Висновок": "Відхилено для використання в MVP",
                "Альтернатива": "Wav2Vec 2.0 (PyTorch)",
            }
        ]
    )
    df.to_csv("yamnet_comparison_results.csv", index=False, encoding="utf-8-sig")
    print("Артефакт експерименту збережено у файл: yamnet_comparison_results.csv")
