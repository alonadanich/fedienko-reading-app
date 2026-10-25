import os
import librosa
import numpy as np
import pandas as pd


def get_mfcc_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    # Відрізаємо тишу
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
    # Відкидаємо гучність
    return mfcc[1:]


def compare_audio_dtw(file_etalon, file_test):
    mfcc1 = get_mfcc_features(file_etalon)
    mfcc2 = get_mfcc_features(file_test)

    # Використовуємо косинусну відстань - вона найстабільніша для MFCC
    D, wp = librosa.sequence.dtw(X=mfcc1, Y=mfcc2, metric="cosine")

    # Сира середня відстань на кожному кроці
    raw_distance = D[-1, -1] / len(wp)

    # Орієнтовна формула (скорегуємо її після результатів)
    similarity = max(0.0, 100.0 - (raw_distance * 150))
    return raw_distance, min(similarity, 100.0)


# ==========================================
# КАЛІБРУВАННЯ ПОРОГУ (THRESHOLD CALIBRATION)
# ==========================================

print("Калібрування алгоритму DTW (Cosine Metric)...\n")

test_pairs = [
    # 1. Позитивні тести (однакові звуки)
    {"звук": "а", "etalon": "test_audio/a_.wav", "child": "test_audio/a.wav"},
    {"звук": "у", "etalon": "test_audio/u_.wav", "child": "test_audio/u.wav"},
    {"звук": "ма", "etalon": "test_audio/ma_.wav", "child": "test_audio/ma.wav"},
    # 2. Негативні тести (Абсолютно різні звуки)
    {
        "звук": "Помилка (А vs У)",
        "etalon": "test_audio/a_.wav",
        "child": "test_audio/u.wav",
    },
    {
        "звук": "Помилка (А vs М)",
        "etalon": "test_audio/a_.wav",
        "child": "test_audio/m.wav",
    },
]

results = []

for pair in test_pairs:
    sound_name = pair["звук"]
    etalon_path = pair["etalon"]
    child_path = pair["child"]

    try:
        raw_dist, score = compare_audio_dtw(etalon_path, child_path)

        print(
            f"[{sound_name}] | Сира дистанція: {raw_dist:.4f} | Орієнтовно: {score:.1f}%"
        )

        results.append(
            {"Звук": sound_name, "Сира дистанція": raw_dist, "Схожість (%)": score}
        )

    except Exception as e:
        print(f"Помилка [{sound_name}]: {e}")
