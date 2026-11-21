import json
import os
import time
from difflib import SequenceMatcher
import joblib
import librosa
import numpy as np
import torch
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Wav2Vec2ForSequenceClassification,
)

# =====================================================================
# 1. ГЛОБАЛЬНІ НАЛАШТУВАННЯ СЕРЕДОВИЩА
# =====================================================================
# Вимикаємо зайві попередження Hugging Face про токен авторизації
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

# Вказуємо бібліотеці Keras 3 працювати поверх вже встановленого PyTorch.
# Це дозволяє відкривати файли .keras на Python 3.14 без встановлення важкого TensorFlow.
os.environ.setdefault("KERAS_BACKEND", "torch")

# Базова папка проєкту (щоб програма завжди знаходила файли незалежно від способу запуску)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "model_config.json")

# Стартова конфігурація трирівневого маршрутизатора за замовчуванням
DEFAULT_CONFIG = {
    # Спільне акустичне ядро ("вуха" системи), яке перетворює звук на числові вектори
    "base_extractor_id": "anton-l/wav2vec2-large-xlsr-53-ukrainian",

    # РІВЕНЬ 1: Модель для окремих літер (1 звук — працює в усіх замочках 1–10)
    # Використовує 8-й шар трансформера, який найкраще фіксує чисту артикуляцію фонеми
    "letters": {
        "type": "sklearn",  # Доступні формати: "sklearn" (.pkl), "keras" (.keras), "hf_seq_clf" (папка HF)
        "layer": 8,
        "model_path": "model_mlp_singles.pkl",
        "scaler_path": "scaler_singles.pkl",
        "encoder_path": "le_singles.pkl",
        "min_conf": 8.0,
    },

    # РІВЕНЬ 2: Модель для двобуквених складів (2 звуки — працює у замочках 2–10)
    # Використовує 12-й шар трансформера, який краще охоплює плавний перехід між двома звуками
    "syllables": {
        "type": "sklearn",  # Доступні формати: "sklearn" (.pkl), "keras" (.keras), "hf_seq_clf" (папка HF)
        "layer": 12,
        "model_path": "model_mlp_pairs.pkl",
        "scaler_path": "scaler_pairs.pkl",
        "encoder_path": "le_pairs.pkl",
        "min_conf": 8.0,
    },

    # РІВЕНЬ 3: Модель для слів та речень (3+ літери — працює у замочках 4–10)
    # Використовує CTC-декодер для посимвольної транскрибації довільних слів та речень
    "words": {
        "type": "ctc",
        "model_path": "anton-l/wav2vec2-large-xlsr-53-ukrainian",
        "similarity_threshold": 0.72,
    },
}

# Таблиця перекладу латинських назв класів (з навчального датасету) в українську кирилицю
LAT_TO_CYR = {
    "a": "а", "u": "у", "o": "о", "y": "и", "i": "і",
    "m": "м", "n": "н", "s": "с", "l": "л", "k": "к",
}

# Акустичні двійники при плавному злитті голосних (коартикуляції) на домашньому мікрофоні:
# наприклад, швидкий перехід губ із вузького [у] в широке [а] утворює призвук [ўа], близький до твердого [ла]
COARTICULATION_ALIASES = {
    "уа": {"ла", "ва"},
    "ао": {"ау", "ал"},
    "ау": {"ао", "ал"},
}


# =====================================================================
# 2. ДОПОМІЖНІ ФУНКЦІЇ (ШЛЯХИ, ВАГА МОДЕЛЕЙ, КОНФІГУРАЦІЯ, ОБРОБКА АУДІО)
# =====================================================================
def resolve_model_path(path_str: str) -> str:
    """
    Перевіряє шлях до файлу або папки моделі:
    - якщо вибрано зовнішній шлях на диску (наприклад, C:/Users/.../model.pkl) — бере його напряму;
    - якщо вказано відносну назву файлу — шукає в папці проєкту;
    - якщо це ідентифікатор Hugging Face Hub ('anton-l/...') — повертає як є для завантаження з кешу HF.
    """
    clean_p = (path_str or "").strip()
    if not clean_p:
        return ""
    if os.path.exists(clean_p):
        return clean_p
    local_candidate = os.path.join(BASE_DIR, clean_p)
    if os.path.exists(local_candidate):
        return local_candidate
    return clean_p


def get_path_size_mb(path_str: str) -> float:
    """Обчислює точну вагу файлу (.pkl, .keras) або цілої папки моделі (HF) у мегабайтах."""
    resolved = resolve_model_path(path_str)
    if not resolved or not os.path.exists(resolved):
        return 0.0
    if os.path.isfile(resolved):
        return os.path.getsize(resolved) / (1024 * 1024)

    total_bytes = 0
    for root, _, files in os.walk(resolved):
        for fname in files:
            fpath = os.path.join(root, fname)
            if os.path.isfile(fpath):
                total_bytes += os.path.getsize(fpath)
    return total_bytes / (1024 * 1024)


def load_config() -> dict:
    """Читає збережені налаштування з model_config.json (без хардкоду шляхів у коді)."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            merged = json.loads(json.dumps(DEFAULT_CONFIG))
            merged["base_extractor_id"] = saved.get("base_extractor_id", merged["base_extractor_id"])
            for tier in ("letters", "syllables", "words"):
                if tier in saved and isinstance(saved[tier], dict):
                    merged[tier].update(saved[tier])
            return merged
        except Exception:
            pass
    save_config(DEFAULT_CONFIG)
    return json.loads(json.dumps(DEFAULT_CONFIG))


def save_config(cfg: dict):
    """Зберігає вибрані користувачем моделі та шари у JSON-файл."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def to_cyrillic(label: str) -> str:
    """Уніфікує назву класу до нижнього регістру української кирилиці."""
    res = str(label).lower().strip()
    for lat, cyr in LAT_TO_CYR.items():
        res = res.replace(lat, cyr)
    return res


def clean_text(text: str) -> str:
    """Прибирає розділові знаки, дефіси складів та допоміжні лінії злиття ('┆') для чистого порівняння."""
    cleaned = text.lower()
    for ch in (" ", "-", "|", "┆", ",", ".", "!", "?", "’", "'"):
        cleaned = cleaned.replace(ch, "")
    return cleaned.strip()


def preprocess_mic_audio(audio_path: str, sr: int = 16000) -> np.ndarray:
    """
    Попередня обробка живого запису з мікрофона:
    1) Зрізає механічне клацання кнопки миші/тачпада на початку (60 мс) і в кінці (100 мс) запису.
    2) М'яко обрізає тишу по краях (top_db=32) із захисним відступом 60 мс,
       щоб не "проковтнути" тихі початкові звуки ('у', 'і', глухі приголосні 'к', 'с').
    3) Нормалізує гучність, щоб модель однаково добре чула і тихий, і гучний дитячий голос.
    """
    speech, _ = librosa.load(audio_path, sr=sr)

    if len(speech) > int(sr * 0.65):
        speech = speech[int(sr * 0.06) : len(speech) - int(sr * 0.10)]

    _, idx = librosa.effects.trim(speech, top_db=32)
    pad = int(sr * 0.06)
    start = max(0, idx[0] - pad)
    end = min(len(speech), idx[1] + pad)
    trimmed = speech[start:end]

    if len(trimmed) > int(sr * 0.12):
        speech = trimmed

    return librosa.util.normalize(speech)


# =====================================================================
# 3. ГОЛОВНЕ ЯДРО РОЗПІЗНАВАННЯ (ТРИРІВНЕВИЙ МАРШРУТИЗАТОР МОДЕЛЕЙ)
# =====================================================================
class ModularSpeechEngine:
    """
    Ієрархічний маршрутизатор розпізнавання мовлення:
      - 1 символ  -> Рівень 1 ('letters'): класифікатор окремих звуків (L8)
      - 2 символи -> Рівень 2 ('syllables'): класифікатор двобуквених складів (L12)
      - 3+ символи -> Рівень 3 ('words'): CTC-декодер слів та речень
    Підтримує гарячу заміну моделей різних фреймворків (.pkl, .keras, Hugging Face) без перезапуску програми.
    """

    def __init__(self):
        self.cfg = load_config()
        self.last_load_seconds = 0.0
        self.feature_extractor = None
        self.wav2vec_base = None
        self.ctc_processor = None
        self.wav2vec_ctc = None
        self.tiers = {"letters": {}, "syllables": {}}
        self.reload_models()

    def _load_tier_bundle(self, tier_cfg: dict) -> dict:
        """Завантажує класифікатор обраного формату (sklearn / keras / hf_seq_clf) та обчислює його вагу."""
        mtype = tier_cfg.get("type", "sklearn")
        mpath = resolve_model_path(tier_cfg.get("model_path", ""))
        sc_path = resolve_model_path(tier_cfg.get("scaler_path", ""))
        enc_path = resolve_model_path(tier_cfg.get("encoder_path", ""))

        bundle = {
            "type": mtype,
            "cfg": tier_cfg,
            "size_mb": get_path_size_mb(mpath),
        }

        # ФОРМАТ 1: Класичний конвеєр Scikit-Learn (.pkl / .joblib)
        if mtype == "sklearn":
            bundle["clf"] = joblib.load(mpath)
            if sc_path and os.path.exists(sc_path):
                bundle["scaler"] = joblib.load(sc_path)
                bundle["size_mb"] += get_path_size_mb(sc_path)
            else:
                bundle["scaler"] = None
            bundle["le"] = joblib.load(enc_path)
            bundle["known_cyr"] = {clean_text(to_cyrillic(c)) for c in bundle["le"].classes_}

        # ФОРМАТ 2: Штатний класифікатор Hugging Face (папка Wav2Vec2ForSequenceClassification)
        elif mtype == "hf_seq_clf":
            hf_model = Wav2Vec2ForSequenceClassification.from_pretrained(mpath)
            hf_model.eval()
            bundle["hf_model"] = hf_model
            bundle["known_cyr"] = {
                clean_text(to_cyrillic(lbl)) for lbl in hf_model.config.id2label.values()
            }

        # ФОРМАТ 3: Нейромережева модель Keras 3 (.keras / .h5 поверх PyTorch)
        # Нормалізація (StandardScaler) вже вбудована у ваги першого шару .keras при експорті
        elif mtype == "keras":
            try:
                import keras
            except ImportError:
                from tensorflow import keras
            bundle["keras_model"] = keras.models.load_model(mpath)
            bundle["le"] = joblib.load(enc_path)
            bundle["known_cyr"] = {clean_text(to_cyrillic(c)) for c in bundle["le"].classes_}

        return bundle

    def reload_models(self) -> tuple[bool, str, float]:
        """
        Перезавантажує всі 3 рівні моделей згідно з поточним model_config.json.
        Якщо базовий екстрактор та CTC-модель збігаються, використовує спільне ядро в пам'яті (Shared Backbone),
        щоб не витрачати вдвічі більше оперативної пам'яті.
        """
        self.cfg = load_config()
        t0 = time.perf_counter()

        base_id = resolve_model_path(self.cfg["base_extractor_id"])
        words_id = resolve_model_path(self.cfg["words"]["model_path"]) or base_id

        # Wav2Vec2FeatureExtractor працює з будь-якою акустичною моделлю Wav2Vec 2.0 (навіть без текстового словника)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_id)

        # Завантажуємо модель Рівня 3 (CTC для слів та речень)
        self.ctc_processor = Wav2Vec2Processor.from_pretrained(words_id)
        self.wav2vec_ctc = Wav2Vec2ForCTC.from_pretrained(words_id)
        self.wav2vec_ctc.eval()

        # Економія пам'яті: перевикористовуємо базовий трансформер, якщо шляхи однакові
        if words_id == base_id:
            self.wav2vec_base = self.wav2vec_ctc.wav2vec2
        else:
            from transformers import Wav2Vec2Model
            self.wav2vec_base = Wav2Vec2Model.from_pretrained(base_id)
            self.wav2vec_base.eval()

        # Завантажуємо прикладні класифікатори для Рівня 1 (літери) та Рівня 2 (склади)
        self.tiers["letters"] = self._load_tier_bundle(self.cfg["letters"])
        self.tiers["syllables"] = self._load_tier_bundle(self.cfg["syllables"])

        self.last_load_seconds = time.perf_counter() - t0
        l_info = f"Літери [{self.cfg['letters']['type']}]: {self.tiers['letters']['size_mb']:.2f} МБ"
        s_info = f"Склади [{self.cfg['syllables']['type']}]: {self.tiers['syllables']['size_mb']:.2f} МБ"
        return True, f"Активовано ({l_info} | {s_info})", self.last_load_seconds

    def _extract_embedding(self, inputs, layer_idx: int, temporal_pool: bool) -> np.ndarray:
        """
        Витягує числовий вектор ознак (ембедінг) із заданого шару (layer_idx) трансформера Wav2Vec 2.0:
          - temporal_pool=False -> середнє значення по всьому звуку (1024 числа, оптимально для 1 літери);
          - temporal_pool=True  -> ділить запис на 3 часові частини: початок, середину й кінець (3072 числа),
            щоб модель чітко розрізняла порядок букв у складі (наприклад, 'ау' проти 'уа').
        """
        with torch.no_grad():
            outputs = self.wav2vec_base(**inputs, output_hidden_states=True)
        feat = outputs.hidden_states[layer_idx].squeeze(0)

        if not temporal_pool:
            return feat.mean(dim=0).numpy()

        if feat.shape[0] < 3:
            feat = feat.repeat(3, 1)
        chunks = torch.tensor_split(feat, 3, dim=0)
        return torch.cat([c.mean(dim=0) for c in chunks]).numpy()

    def _predict_ranked(self, tier_name: str, inputs) -> list[tuple[str, float]]:
        """
        Проганяє звук через обраний класифікатор (sklearn / keras / hf_seq_clf)
        і повертає повний рейтинг гіпотез від найвищої ймовірності до найнижчої: [('а', 99.4), ...].
        """
        bundle = self.tiers[tier_name]
        tcfg = bundle["cfg"]
        mtype = bundle["type"]
        layer_idx = int(tcfg.get("layer", 12))

        # Варіант А: Штатна папка Hugging Face (Wav2Vec2ForSequenceClassification)
        if mtype == "hf_seq_clf":
            with torch.no_grad():
                logits = bundle["hf_model"](**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            id2lbl = bundle["hf_model"].config.id2label
            labels = [
                clean_text(to_cyrillic(id2lbl.get(i, id2lbl.get(str(i), str(i)))))
                for i in range(len(probs))
            ]

        # Варіант Б: Модель Keras (.keras)
        elif mtype == "keras":
            km = bundle["keras_model"]
            if getattr(km, "input_shape", None) is not None:
                in_dim = int(km.input_shape[-1])
            elif getattr(km, "inputs", None):
                in_dim = int(km.inputs[0].shape[-1])
            else:
                in_dim = int(km.layers[0].kernel.shape[0])

            emb = self._extract_embedding(inputs, layer_idx, temporal_pool=(in_dim == 3072))
            x_in = np.expand_dims(emb.astype(np.float32), axis=0)
            probs = np.asarray(km.predict(x_in, verbose=0))[0]
            labels = [clean_text(to_cyrillic(c)) for c in bundle["le"].classes_]

        # Варіант В: Модель Scikit-Learn (.pkl)
        else:
            ref_obj = bundle["scaler"] if bundle["scaler"] is not None else bundle["clf"]
            expected_dim = getattr(ref_obj, "n_features_in_", 1024)
            emb = self._extract_embedding(inputs, layer_idx, temporal_pool=(expected_dim == 3072))
            X_in = bundle["scaler"].transform([emb]) if bundle["scaler"] is not None else [emb]
            probs = bundle["clf"].predict_proba(X_in)[0]
            labels = [clean_text(to_cyrillic(c)) for c in bundle["le"].classes_]

        order = np.argsort(probs)[::-1]
        return [(labels[i], float(probs[i] * 100.0)) for i in order]

    def _decode_ctc_raw(self, speech: np.ndarray) -> str:
        """Транскрибує аудіо по літерах за допомогою штатної мовної моделі Wav2Vec2ForCTC."""
        ctc_inputs = self.ctc_processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.wav2vec_ctc(**ctc_inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]
        decoded = self.ctc_processor.tokenizer.decode(pred_ids, output_char_offsets=True)
        return clean_text(decoded.text)

    def _predict_ctc_words(self, speech: np.ndarray, acceptable_cyr: set[str], t_start: float) -> tuple[bool, str, float]:
        """Оцінює вимову довгих слів та речень (Рівень 3) за коефіцієнтом схожості рядків."""
        heard_clean = self._decode_ctc_raw(speech)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        print(f"🎙️ [Слово/Речення] Ціль: {acceptable_cyr} | CTC почув: '{heard_clean}' | Час: {elapsed_ms:.0f} мс")

        if not heard_clean:
            return False, "тиша", 0.0

        for target_opt in acceptable_cyr:
            if target_opt in heard_clean:
                return True, heard_clean, 95.0

        best_sim = max(SequenceMatcher(None, opt, heard_clean).ratio() for opt in acceptable_cyr)
        thresh = float(self.cfg["words"].get("similarity_threshold", 0.72))
        return best_sim >= thresh, heard_clean, float(best_sim * 100)

    def evaluate(self, audio_path: str, target_code: str, alt_targets: list[str] | None = None):
        """
        Головна точка входу, яку викликає тренажер після натискання кнопки 'Стоп' на мікрофоні.
        Автоматично визначає довжину цілі та направляє запис у потрібну модель.
        """
        t_start = time.perf_counter()
        speech = preprocess_mic_audio(audio_path, sr=16000)

        # Формуємо множину допустимих правильних відповідей
        target_cyr = clean_text(to_cyrillic(target_code))
        acceptable = {target_cyr}
        if alt_targets:
            acceptable.update(clean_text(to_cyrillic(t)) for t in alt_targets)
        acceptable.update(COARTICULATION_ALIASES.get(target_cyr, set()))

        # Дзеркальний склад для захисту від читання задом наперед ('ау' замість 'уа')
        reversed_target = target_cyr[::-1] if len(target_cyr) == 2 and target_cyr[0] != target_cyr[1] else ""

        chars_len = len(target_cyr)
        inputs = self.feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)

        # Вибір маршруту: 1 буква -> 'letters', 2 букви -> 'syllables', 3+ букв -> CTC
        tier_name = None
        if chars_len == 1:
            tier_name = "letters"
        elif chars_len == 2 and target_cyr in self.tiers["syllables"].get("known_cyr", set()):
            tier_name = "syllables"

        if tier_name is not None:
            ranked = self._predict_ranked(tier_name, inputs)
            top1_lbl, top1_conf = ranked[0]
            min_c = float(self.cfg[tier_name].get("min_conf", 8.0))
            ctc_heard = self._decode_ctc_raw(speech)
            elapsed_ms = (time.perf_counter() - t_start) * 1000

            top3_log = ", ".join(f"{lbl}:{c:.1f}%" for lbl, c in ranked[:3])
            print(f"🎙️ [{tier_name}] Ціль: '{target_cyr}' | Топ-3: [{top3_log}] | CTC: '{ctc_heard}' | Час: {elapsed_ms:.0f} мс")

            # Перевірка 1: Топ-1 гіпотеза збігається з ціллю
            if top1_lbl in acceptable and top1_conf >= min_c:
                return True, target_cyr, top1_conf

            # Блокування реверсу: якщо дитина прочитала склад справа наліво — одразу відхиляємо
            if reversed_target and (top1_lbl == reversed_target or ctc_heard == reversed_target):
                return False, top1_lbl, top1_conf

            # Перевірка 2: Ціль є серед Топ-3 найімовірніших класів із вагомою часткою впевненості
            for _, (cand_lbl, cand_conf) in enumerate(ranked[:3]):
                if cand_lbl in acceptable and cand_conf >= max(6.0, top1_conf * 0.35):
                    return True, cand_lbl, max(cand_conf, top1_conf)

            # Перевірка 3: Резервне підтвердження від CTC-моделі при дуже чіткій вимові
            if ctc_heard and (ctc_heard == target_cyr or (len(target_cyr) == 2 and target_cyr in ctc_heard)):
                return True, ctc_heard, 90.0

            return False, top1_lbl, top1_conf

        # МАРШРУТ 3: Слова з 3+ літер та речення
        return self._predict_ctc_words(speech, acceptable, t_start)