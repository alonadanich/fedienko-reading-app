import os
import warnings
import joblib
import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2ForCTC,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
)

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

MODEL_ID = "anton-l/wav2vec2-large-xlsr-53-ukrainian"
DATA_DIR = "dataset_augmented"
EXP_CACHE = "hf_ctc_and_layers_cache.pkl"
LAYERS_CACHE = "features_layers_cache.pkl"
ULTIMATE_CACHE = "ultimate_pairs_cache.pkl"

LAT_TO_CYR = {
    "a": "а", "u": "у", "o": "о", "y": "и", "i": "і",
    "m": "м", "n": "н", "s": "с", "l": "л", "k": "к",
}


def to_cyrillic(text: str) -> str:
    res = str(text).lower().strip()
    for lat, cyr in LAT_TO_CYR.items():
        res = res.replace(lat, cyr)
    return res


def clean_ctc_text(text: str) -> str:
    cleaned = text.lower()
    for ch in (" ", "-", "|", "┆", ",", ".", "!", "?", "’", "'"):
        cleaned = cleaned.replace(ch, "")
    return cleaned.strip()


class ExactHFSequenceHead(nn.Module):
    """Точна архітектура класифікаційної голови Wav2Vec2ForSequenceClassification із бібліотеки transformers."""

    def __init__(self, num_labels: int, use_weighted_sum: bool = False, proj_size: int = 256, num_layers: int = 25):
        super().__init__()
        self.use_weighted_sum = use_weighted_sum
        if use_weighted_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.dropout = nn.Dropout(0.1)
        self.projector = nn.Linear(1024, proj_size)
        self.classifier = nn.Linear(proj_size, num_labels)

    def forward(self, x_layers: torch.Tensor) -> torch.Tensor:
        if self.use_weighted_sum:
            norm_w = torch.softmax(self.layer_weights, dim=-1)
            h = (x_layers * norm_w.view(1, -1, 1)).sum(dim=1)
        else:
            h = x_layers[:, -1, :]  # Штатний останній (24-й) шар Wav2Vec 2.0
        h = self.dropout(h)
        return self.classifier(self.projector(h))


def eval_hf_head_cv(X_all_layers: np.ndarray, y_enc: np.ndarray, groups: np.ndarray,
                    num_classes: int, trained: bool, use_weighted_sum: bool, epochs: int = 35) -> float:
    """Оцінює штатну голову HF Wav2Vec2ForSequenceClassification на чесній 5-фолдовій GroupKFold."""
    gkf = GroupKFold(n_splits=5)
    fold_scores = []

    for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X_all_layers, y_enc, groups)):
        torch.manual_seed(42 + fold_idx)
        head = ExactHFSequenceHead(num_labels=num_classes, use_weighted_sum=use_weighted_sum)

        X_te_t = torch.tensor(X_all_layers[te_idx], dtype=torch.float32)

        if trained:
            head.train()
            X_tr_t = torch.tensor(X_all_layers[tr_idx], dtype=torch.float32)
            y_tr_t = torch.tensor(y_enc[tr_idx], dtype=torch.long)

            ds = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
            loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

            optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            criterion = nn.CrossEntropyLoss()

            for _ in range(epochs):
                for bx, by in loader:
                    optimizer.zero_grad()
                    loss = criterion(head(bx), by)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

        head.eval()
        with torch.no_grad():
            preds = torch.argmax(head(X_te_t), dim=-1).numpy()
        fold_scores.append(accuracy_score(y_enc[te_idx], preds))

    return float(np.mean(fold_scores) * 100.0)


def save_trained_hf_singles_folder(X_singles_layers: np.ndarray, y_singles: np.ndarray):
    """Зберігає реально навчену модель Wav2Vec2ForSequenceClassification у папку external_models/hf_seq_singles."""
    out_dir = os.path.join("external_models", "hf_seq_singles")
    le = LabelEncoder()
    y_enc = le.fit_transform(y_singles)
    num_classes = len(le.classes_)

    torch.manual_seed(42)
    head = ExactHFSequenceHead(num_labels=num_classes, use_weighted_sum=True)
    head.train()
    ds = torch.utils.data.TensorDataset(
        torch.tensor(X_singles_layers, dtype=torch.float32),
        torch.tensor(y_enc, dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    for _ in range(35):
        for bx, by in loader:
            opt.zero_grad()
            crit(head(bx), by).backward()
            opt.step()
    head.eval()

    id2label = {int(i): str(lbl) for i, lbl in enumerate(le.classes_)}
    label2id = {str(lbl): int(i) for i, lbl in enumerate(le.classes_)}
    cfg = Wav2Vec2Config.from_pretrained(
        MODEL_ID,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        use_weighted_layer_sum=True,
        classifier_proj_size=256,
    )
    hf_model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID, config=cfg)
    with torch.no_grad():
        hf_model.layer_weights.copy_(head.layer_weights)
        hf_model.projector.weight.copy_(head.projector.weight)
        hf_model.projector.bias.copy_(head.projector.bias)
        hf_model.classifier.weight.copy_(head.classifier.weight)
        hf_model.classifier.bias.copy_(head.classifier.bias)

    os.makedirs(out_dir, exist_ok=True)
    hf_model.save_pretrained(out_dir)
    print(f"✅ Навчену штатну модель HF збережено у '{out_dir}' (можна тестувати через HF Folder у UI!).")


def build_or_load_exp_cache():
    if os.path.exists(EXP_CACHE):
        print(f"📁 Завантаження готового кешу експериментів із '{EXP_CACHE}'...")
        return joblib.load(EXP_CACHE)

    print("⏳ Перший запуск: одноразовий прохід по dataset_augmented для CTC та всіх 25 шарів (~10-12 хв)...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    ctc_model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    ctc_model.eval()

    cache = {
        "singles": {"layers_mean": [], "temporal_12": [], "y": [], "groups": [], "ctc_pred": []},
        "pairs": {"layers_mean": [], "temporal_12": [], "y": [], "groups": [], "ctc_pred": []},
    }

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]
    for i, fname in enumerate(files):
        parts = fname.split("_")
        if len(parts) >= 3:
            speaker_id = parts[0].lower()
            label = parts[1].lower()
            cat = "singles" if len(label) == 1 else "pairs"
            fpath = os.path.join(DATA_DIR, fname)

            try:
                speech, _ = librosa.load(fpath, sr=16000)
                speech, _ = librosa.effects.trim(speech, top_db=20)
                speech = librosa.util.normalize(speech)

                inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
                with torch.no_grad():
                    out = ctc_model(**inputs, output_hidden_states=True)

                # 1. Прогноз штатного CTC-декодера "з коробки"
                pred_ids = torch.argmax(out.logits, dim=-1)[0]
                ctc_str = clean_ctc_text(processor.tokenizer.decode(pred_ids))

                # 2. Середні вектори всіх 25 шарів (0..24) для Wav2Vec2ForSequenceClassification
                all_means = torch.stack([hs.squeeze(0).mean(dim=0) for hs in out.hidden_states], dim=0).numpy()

                # 3. Temporal Pooling (3 часові частини) на 12-му шарі для складів
                f12 = out.hidden_states[12].squeeze(0)
                if f12.shape[0] < 3:
                    f12 = f12.repeat(3, 1)
                t12 = torch.cat([c.mean(dim=0) for c in torch.tensor_split(f12, 3, dim=0)]).numpy()

                cache[cat]["layers_mean"].append(all_means.astype(np.float32))
                cache[cat]["temporal_12"].append(t12.astype(np.float32))
                cache[cat]["y"].append(label)
                cache[cat]["groups"].append(speaker_id)
                cache[cat]["ctc_pred"].append(ctc_str)
            except Exception:
                pass

        if (i + 1) % 200 == 0:
            print(f"   Оброблено {i + 1} / {len(files)} аудіофайлів...")

    for cat in ("singles", "pairs"):
        cache[cat]["layers_mean"] = np.array(cache[cat]["layers_mean"], dtype=np.float32)
        cache[cat]["temporal_12"] = np.array(cache[cat]["temporal_12"], dtype=np.float32)
        cache[cat]["y"] = np.array(cache[cat]["y"])
        cache[cat]["groups"] = np.array(cache[cat]["groups"])
        cache[cat]["ctc_pred"] = np.array(cache[cat]["ctc_pred"])

    joblib.dump(cache, EXP_CACHE)
    print(f"✅ Кеш збережено у '{EXP_CACHE}'. Повторні запуски триватимуть секунди!\n")
    return cache


def eval_sklearn_cv(clf_template, X: np.ndarray, y_enc: np.ndarray, groups: np.ndarray) -> float:
    gkf = GroupKFold(n_splits=5)
    scores = []
    for tr_idx, te_idx in gkf.split(X, y_enc, groups):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr_idx])
        X_te = sc.transform(X[te_idx])
        model = clone(clf_template)
        model.fit(X_tr, y_enc[tr_idx])
        scores.append(accuracy_score(y_enc[te_idx], model.predict(X_te)))
    return float(np.mean(scores) * 100.0)


def main():
    cache = build_or_load_exp_cache()

    s_layers = cache["singles"]["layers_mean"]
    s_y = cache["singles"]["y"]
    s_grp = cache["singles"]["groups"]
    s_ctc = cache["singles"]["ctc_pred"]

    p_layers = cache["pairs"]["layers_mean"]
    p_temp12 = cache["pairs"]["temporal_12"]
    p_y = cache["pairs"]["y"]
    p_grp = cache["pairs"]["groups"]
    p_ctc = cache["pairs"]["ctc_pred"]

    le_s = LabelEncoder()
    y_s_enc = le_s.fit_transform(s_y)
    le_p = LabelEncoder()
    y_p_enc = le_p.fit_transform(p_y)

    print(f"📊 Датасет: Дикторів = {len(np.unique(np.concatenate([s_grp, p_grp])))} | "
          f"Одиничні звуки: {len(s_y)} файлів ({len(le_s.classes_)} класів) | "
          f"Склади: {len(p_y)} файлів ({len(le_p.classes_)} класів)\n")

    # 1. Штатний Wav2Vec2ForCTC без донавчання
    s_y_cyr = np.array([to_cyrillic(lbl) for lbl in s_y])
    p_y_cyr = np.array([to_cyrillic(lbl) for lbl in p_y])
    ctc_s_exact = float(np.mean(s_ctc == s_y_cyr) * 100.0)
    ctc_p_exact = float(np.mean(p_ctc == p_y_cyr) * 100.0)

    # 2. Штатний Wav2Vec2ForSequenceClassification без донавчання ("з коробки")
    hf_zero_s = eval_hf_head_cv(s_layers, y_s_enc, s_grp, len(le_s.classes_), trained=False, use_weighted_sum=False)
    hf_zero_p = eval_hf_head_cv(p_layers, y_p_enc, p_grp, len(le_p.classes_), trained=False, use_weighted_sum=False)

    # 3. Штатний Wav2Vec2ForSequenceClassification (донавчання голови, шар L24 за замовчуванням)
    print("🔬 Навчання штатної голови HF SequenceClassification (L24, 5 фолдів)...")
    hf_l24_s = eval_hf_head_cv(s_layers, y_s_enc, s_grp, len(le_s.classes_), trained=True, use_weighted_sum=False)
    hf_l24_p = eval_hf_head_cv(p_layers, y_p_enc, p_grp, len(le_p.classes_), trained=True, use_weighted_sum=False)

    # 4. Штатний Wav2Vec2ForSequenceClassification (донавчання + use_weighted_layer_sum=True)
    print("🔬 Навчання штатної голови HF SequenceClassification (Weighted 25 Layers, 5 фолдів)...")
    hf_wsum_s = eval_hf_head_cv(s_layers, y_s_enc, s_grp, len(le_s.classes_), trained=True, use_weighted_sum=True)
    hf_wsum_p = eval_hf_head_cv(p_layers, y_p_enc, p_grp, len(le_p.classes_), trained=True, use_weighted_sum=True)

    # 5. Класичний SVM (RBF) на оптимальних шарах (L8 для звуків, L12 + Temporal Pooling для складів)
    print("🔬 Перевірка SVM (RBF) та цільових моделей MLP на тих самих 5 фолдах...")
    svm_tpl = SVC(kernel="rbf", C=10.0, class_weight="balanced", random_state=42)
    svm_s = eval_sklearn_cv(svm_tpl, s_layers[:, 8, :], y_s_enc, s_grp)
    svm_p = eval_sklearn_cv(svm_tpl, p_temp12, y_p_enc, p_grp)

    # 6. Ваші робочі моделі MLP із застосунку "Капітошка"
    mlp_s_tpl = joblib.load("model_mlp_singles.pkl") if os.path.exists("model_mlp_singles.pkl") else MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=42)
    mlp_p_tpl = joblib.load("model_mlp_pairs.pkl") if os.path.exists("model_mlp_pairs.pkl") else MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=300, random_state=42)
    mlp_s = eval_sklearn_cv(mlp_s_tpl, s_layers[:, 8, :], y_s_enc, s_grp)
    mlp_p = eval_sklearn_cv(mlp_p_tpl, p_temp12, y_p_enc, p_grp)

    # Додаткова перевірка ultimate_pairs_cache.pkl (якщо є)
    ult_p_str = "—"
    if os.path.exists(ULTIMATE_CACHE):
        X_u, y_u, g_u = joblib.load(ULTIMATE_CACHE)
        y_u_enc = LabelEncoder().fit_transform(y_u)
        ult_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=512, random_state=42)),
            ("mlp", MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation="tanh", alpha=1.0, early_stopping=True, random_state=42)),
        ])
        gkf = GroupKFold(n_splits=5)
        u_scores = [
            accuracy_score(y_u_enc[te], ult_pipe.fit(X_u[tr], y_u_enc[tr]).predict(X_u[te]))
            for tr, te in gkf.split(X_u, y_u_enc, g_u)
        ]
        ult_p_str = f"{np.mean(u_scores) * 100:.2f}%"

    # Оновлюємо папку external_models/hf_seq_singles реальними навченими вагами
    save_trained_hf_singles_folder(s_layers, s_y)

    print("\n" + "=" * 95)
    print("📊 ЗВЕДЕНА ТАБЛИЦЯ 4.2 (SPEAKER-INDEPENDENT, GROUPKFOLD = 5 ФОЛДІВ)")
    print("=" * 95)
    print("| Архітектура / конфігурація моделі | Одиничні звуки (%) | Двобуквені склади (%) | Вага голови / файлу |")
    print("| :--- | :---: | :---: | :---: |")
    print(f"| 1. Штатний `Wav2Vec2ForCTC` (без донавчання, «з коробки») | {ctc_s_exact:.2f}% | {ctc_p_exact:.2f}% | 1205.00 МБ (спільне ядро) |")
    print(f"| 2. Штатний `Wav2Vec2ForSequenceClassification` (без донавчання) | {hf_zero_s:.2f}% | {hf_zero_p:.2f}% | 1204.37 МБ |")
    print(f"| 3. Штатний `Wav2Vec2ForSequenceClassification` (донавчання, шар L24) | {hf_l24_s:.2f}% | {hf_l24_p:.2f}% | 1204.37 МБ |")
    print(f"| 4. Штатний `Wav2Vec2ForSequenceClassification` (`use_weighted_layer_sum=True`) | {hf_wsum_s:.2f}% | {hf_wsum_p:.2f}% | 1204.37 МБ |")
    print(f"| 5. Кастомний конвеєр `Wav2Vec 2.0 + SVM (RBF)` (L8 / L12 Temporal) | {svm_s:.2f}% | {svm_p:.2f}% | ~35–65 МБ (`.pkl`) |")
    print(f"| 6. **Запропонований конвеєр `Wav2Vec 2.0 + MLP` (L8 / L12 Temporal)** | **{mlp_s:.2f}%** | **{mlp_p:.2f}%** | **1.03 МБ + 6.68 МБ (`.keras`)** |")
    if ult_p_str != "—":
        print(f"| *(Додатково)* `Multi-Layer (L12+L16) + PCA + MLP` | — | {ult_p_str} | ~12 МБ |")
    print("=" * 95)


if __name__ == "__main__":
    main()