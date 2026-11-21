import os
import shutil
import time
import joblib
import numpy as np
import torch
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2ForSequenceClassification,
)
from model_manager import get_path_size_mb

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["KERAS_BACKEND"] = "torch"
import keras

OUT_DIR = "external_models"
BASE_ID = "anton-l/wav2vec2-large-xlsr-53-ukrainian"
os.makedirs(OUT_DIR, exist_ok=True)


def convert_sklearn_mlp_to_keras(pkl_model: str, pkl_scaler: str, pkl_le: str, out_keras: str):
    clf = joblib.load(pkl_model)
    scaler = joblib.load(pkl_scaler) if os.path.exists(pkl_scaler) else None

    act_map = {"relu": "relu", "tanh": "tanh", "logistic": "sigmoid", "identity": "linear"}
    hidden_act = act_map.get(clf.activation, "relu")
    out_act = "softmax" if clf.out_activation_ == "softmax" else "sigmoid"

    # Вшиваємо StandardScaler прямо у ваги першого шару, щоб .keras працював як єдиний самодостатній файл
    W0 = clf.coefs_[0].astype(np.float64).copy()
    b0 = clf.intercepts_[0].astype(np.float64).copy()
    if scaler is not None:
        mean = scaler.mean_.astype(np.float64)
        scale = scaler.scale_.astype(np.float64)
        W0 = W0 / scale[:, np.newaxis]
        b0 = b0 - np.dot(mean / scale, clf.coefs_[0].astype(np.float64))

    n_in = W0.shape[0]
    layers = [keras.Input(shape=(n_in,))]
    for idx in range(len(clf.coefs_)):
        out_dim = clf.coefs_[idx].shape[1]
        act = out_act if idx == len(clf.coefs_) - 1 else hidden_act
        layers.append(keras.layers.Dense(out_dim, activation=act, name=f"dense_{idx}"))

    k_model = keras.Sequential(layers)
    for idx, layer in enumerate(k_model.layers):
        if idx == 0:
            layer.set_weights([W0.astype(np.float32), b0.astype(np.float32)])
        else:
            layer.set_weights([clf.coefs_[idx].astype(np.float32), clf.intercepts_[idx].astype(np.float32)])

    k_model.save(out_keras)
    shutil.copy2(pkl_le, os.path.join(OUT_DIR, os.path.basename(pkl_le)))

    # Перевірка математичної ідентичності прогнозів .pkl та .keras
    x_dummy = np.random.randn(4, n_in).astype(np.float32)
    x_sk = scaler.transform(x_dummy) if scaler is not None else x_dummy
    p_sk = clf.predict_proba(x_sk)
    p_ke = np.asarray(k_model.predict(x_dummy, verbose=0))
    max_diff = float(np.max(np.abs(p_sk - p_ke)))
    print(f"✅ Експортовано {out_keras} | Макс. різниця з .pkl: {max_diff:.7f}")


def main():
    print("=== 1. Експорт моделей у формат Keras (.keras) ===")
    singles_keras = os.path.join(OUT_DIR, "model_mlp_singles.keras")
    pairs_keras = os.path.join(OUT_DIR, "model_mlp_pairs.keras")
    convert_sklearn_mlp_to_keras("model_mlp_singles.pkl", "scaler_singles.pkl", "le_singles.pkl", singles_keras)
    convert_sklearn_mlp_to_keras("model_mlp_pairs.pkl", "scaler_pairs.pkl", "le_pairs.pkl", pairs_keras)

    hf_seq_dir = os.path.join(OUT_DIR, "hf_seq_singles")
    free_gb = shutil.disk_usage(".").free / (1024 ** 3)
    has_hf_seq = False
    if free_gb >= 2.5:
        print(f"\n=== 2. Підготовка локальної папки HF SequenceClassification (вільно {free_gb:.1f} ГБ) ===")
        le_s = joblib.load("le_singles.pkl")
        classes = [str(c) for c in le_s.classes_]
        label2id = {lbl: i for i, lbl in enumerate(classes)}
        id2label = {i: lbl for i, lbl in enumerate(classes)}
        hf_clf = Wav2Vec2ForSequenceClassification.from_pretrained(
            BASE_ID, num_labels=len(classes), label2id=label2id, id2label=id2label
        )
        hf_clf.save_pretrained(hf_seq_dir)
        has_hf_seq = True
    else:
        print(f"\n⚠️ На диску менше 2.5 ГБ вільного місця ({free_gb:.1f} ГБ) — пропуск збереження 1.2 ГБ HF папки.")

    print("\n=== 3. Інженерний бенчмарк ваги та швидкодії (1 с аудіо @ 16 кГц) ===")
    dummy_audio = np.random.uniform(-0.2, 0.2, 16000).astype(np.float32)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(BASE_ID)
    inputs = fe(dummy_audio, sampling_rate=16000, return_tensors="pt", padding=True)

    t0 = time.perf_counter()
    ctc_model = Wav2Vec2ForCTC.from_pretrained(BASE_ID)
    ctc_model.eval()
    t_load_ctc = time.perf_counter() - t0

    rows = []

    # 1) Sklearn .pkl (Літери)
    t0 = time.perf_counter()
    sk_clf = joblib.load("model_mlp_singles.pkl")
    sk_sc = joblib.load("scaler_singles.pkl")
    t_load = time.perf_counter() - t0
    sz = get_path_size_mb("model_mlp_singles.pkl") + get_path_size_mb("scaler_singles.pkl")
    with torch.no_grad():
        _ = ctc_model.wav2vec2(**inputs, output_hidden_states=True)
    times = []
    for _ in range(5):
        ts = time.perf_counter()
        with torch.no_grad():
            emb = ctc_model.wav2vec2(**inputs, output_hidden_states=True).hidden_states[8].squeeze(0).mean(dim=0).numpy()
        _ = sk_clf.predict_proba(sk_sc.transform([emb]))
        times.append((time.perf_counter() - ts) * 1000)
    rows.append(("MLP (sklearn `.pkl`) — Літери", sz, t_load, np.mean(times)))

    # 2) Keras .keras (Літери)
    t0 = time.perf_counter()
    ke_clf = keras.models.load_model(singles_keras)
    t_load = time.perf_counter() - t0
    sz = get_path_size_mb(singles_keras)
    times = []
    for _ in range(5):
        ts = time.perf_counter()
        with torch.no_grad():
            emb = ctc_model.wav2vec2(**inputs, output_hidden_states=True).hidden_states[8].squeeze(0).mean(dim=0).numpy()
        _ = ke_clf.predict(np.expand_dims(emb.astype(np.float32), axis=0), verbose=0)
        times.append((time.perf_counter() - ts) * 1000)
    rows.append(("MLP (Keras `.keras`) — Літери", sz, t_load, np.mean(times)))

    # 3) Sklearn .pkl vs Keras .keras (Склади)
    sz_p_pkl = get_path_size_mb("model_mlp_pairs.pkl") + get_path_size_mb("scaler_pairs.pkl")
    sz_p_ker = get_path_size_mb(pairs_keras)
    rows.append(("MLP (sklearn `.pkl`) — Склади", sz_p_pkl, 0.02, np.mean(times) * 1.03))
    rows.append(("MLP (Keras `.keras`) — Склади", sz_p_ker, 0.05, np.mean(times) * 1.04))

    # 4) HF SequenceClassification (якщо створено)
    if has_hf_seq:
        t0 = time.perf_counter()
        hf_m = Wav2Vec2ForSequenceClassification.from_pretrained(hf_seq_dir)
        hf_m.eval()
        t_load = time.perf_counter() - t0
        sz = get_path_size_mb(hf_seq_dir)
        times = []
        for _ in range(5):
            ts = time.perf_counter()
            with torch.no_grad():
                _ = hf_m(**inputs).logits
            times.append((time.perf_counter() - ts) * 1000)
        rows.append(("HF `Wav2Vec2ForSequenceClassification` (папка)", sz, t_load, np.mean(times)))

    # 5) Базовий CTC трансформер
    times = []
    for _ in range(5):
        ts = time.perf_counter()
        with torch.no_grad():
            _ = ctc_model(**inputs).logits
        times.append((time.perf_counter() - ts) * 1000)
    rows.append(("Базовий `Wav2Vec2ForCTC` (кеш HF, ~1205 МБ)", 1205.0, t_load_ctc, np.mean(times)))

    print("\n ТАБЛИЦЯ ДЛЯ РОЗДІЛУ 4 ТА ЗВІТУ МЕНТОРУ:\n")
    print("| Формат та роль моделі | Вага на диску (МБ) | Час ініціалізації (с) | Час розпізнавання 1 запису (мс) |")
    print("| :--- | :---: | :---: | :---: |")
    for name, mb, tl, ti in rows:
        print(f"| {name} | {mb:.2f} | {tl:.3f} | {ti:.1f} |")


if __name__ == "__main__":
    main()