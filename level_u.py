import flet as ft
import librosa
import torch
import joblib
import numpy as np
import time
import os
import threading
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# ==========================================
# ІНІЦІАЛІЗАЦІЯ МОДЕЛЕЙ (Тимчасово залишаємо тут для тестування рівня окремо)
# ==========================================
print("Завантаження моделей... Зачекайте.")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
processor = Wav2Vec2Processor.from_pretrained("anton-l/wav2vec2-large-xlsr-53-ukrainian")
wav2vec_model = Wav2Vec2Model.from_pretrained("anton-l/wav2vec2-large-xlsr-53-ukrainian")

svm_clf = joblib.load("fedienko_svm_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ДОДАЄМО SILERO VAD (Нейромережевий детектор голосу)
print("Завантаження VAD детектора...")
vad_model, vad_utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = vad_utils


def predict_phoneme(audio_path):
    # 1. Читаємо аудіо за допомогою надійної librosa (замість torchaudio)
    speech_array, rate = librosa.load(audio_path, sr=16000)
    
    # Перетворюємо numpy-масив у PyTorch Tensor для моделі VAD
    wav = torch.from_numpy(speech_array)
    
    # 2. Знаходимо часові мітки (відрізки), де звучить САМЕ ГОЛОС
    speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000)
    
    # 3. Якщо голос знайдено - склеюємо лише корисні шматки
    if len(speech_timestamps) > 0:
        speech = collect_chunks(speech_timestamps, wav).squeeze().numpy()
    else:
        # Якщо голосу не знайдено (напр., суцільна тиша), беремо оригінал
        speech = wav.squeeze().numpy()
        
    # 4. Нормалізуємо гучність
    speech = librosa.util.normalize(speech)
    
    # 5. Пропускаємо через Wav2Vec (12-й шар)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = wav2vec_model(**inputs, output_hidden_states=True)
        features = outputs.hidden_states[12]
        
    features_mean = features.mean(dim=1).numpy()
    probs = svm_clf.predict_proba(features_mean)[0]
    max_index = np.argmax(probs)
    return label_encoder.inverse_transform([max_index])[0], probs[max_index] * 100


def main(page: ft.Page):
    page.title = "Тренажер читання Федієнка - Рівень У"
    page.padding = 20
    page.theme_mode = ft.ThemeMode.LIGHT

    audio_rec = ft.AudioRecorder(audio_encoder=ft.AudioEncoder.WAV)
    page.overlay.append(audio_rec)

    kapi_voice = ft.Audio(autoplay=True)
    page.overlay.append(kapi_voice)

    state = {
        "recording": False,
        "can_record": False,
        "reading_mode": False,
        "step": 0,
        "current_target": "",  # Зберігаємо цільову фонему для перевірки
    }

    # ==========================================
    # ВІДЖЕТИ АНІМАЦІЇ ТА КАПІТОШКИ
    # ==========================================
    balloons = [
        ft.Container(
            content=ft.Image(src="balloon.png", width=80),
            left=50,
            top=800,
            opacity=0,
            animate_position=ft.animation.Animation(3000, ft.AnimationCurve.EASE_OUT),
            animate_opacity=300,
        ),
        ft.Container(
            content=ft.Image(src="balloon.png", width=110),
            left=200,
            top=800,
            opacity=0,
            animate_position=ft.animation.Animation(3500, ft.AnimationCurve.EASE_OUT),
            animate_opacity=300,
        ),
    ]

    kapitoshka_img = ft.Image(
        src="kapitoshka_idle.png", width=150, height=150, fit=ft.ImageFit.CONTAIN
    )
    dialogue_text = ft.Text("", size=18, color=ft.Colors.BLACK87)
    dialogue_bubble = ft.Container(
        content=dialogue_text,
        bgcolor=ft.Colors.LIGHT_BLUE_50,
        padding=15,
        border_radius=15,
        border=ft.border.all(2, ft.Colors.LIGHT_BLUE_200),
        width=450,
    )
    kapitoshka_ui = ft.Row(
        [kapitoshka_img, dialogue_bubble], alignment=ft.MainAxisAlignment.CENTER
    )

    def kapi_speak(text, audio_file=None, face="idle"):
        dialogue_text.value = text

        # Функція для простої зміни обличчя
        def set_face(f):
            if f == "u":
                kapitoshka_img.src = "kapitoshka_u.png"
            elif f == "a":
                kapitoshka_img.src = "kapitoshka_a.png"
            elif f == "sad":
                kapitoshka_img.src = "kapitoshka_sad.png"
            else:
                kapitoshka_img.src = "kapitoshka_idle.png"
            page.update()

        # Якщо це склад - робимо мікро-анімацію!
        if face == "au":
            # Запускаємо в окремому потоці, щоб не блокувати інтерфейс
            def animate_au():
                set_face("a")
                time.sleep(0.6)  # Тримаємо "А"
                set_face("u")
                time.sleep(0.8)  # Переходимо в "У"
                set_face("idle")

            threading.Thread(target=animate_au, daemon=True).start()

        elif face == "ua":

            def animate_ua():
                set_face("u")
                time.sleep(0.6)  # Тримаємо "У"
                set_face("a")
                time.sleep(0.8)  # Переходимо в "А"
                set_face("idle")

            threading.Thread(target=animate_ua, daemon=True).start()

        else:
            # Звичайна статична картинка для одиночних звуків
            set_face(face)

        if audio_file and os.path.exists(os.path.join("assets", audio_file)):
            kapi_voice.src = audio_file
            kapi_voice.update()
        page.update()

    def trigger_success():
        for b in balloons:
            b.top = -150
            b.opacity = 1
        page.update()
        time.sleep(3.0)
        for b in balloons:
            b.opacity = 0
            b.top = 800
        page.update()

    # ==========================================
    # ДИНАМІЧНИЙ ЕКРАН РІВНЯ
    # ==========================================
    level_content = ft.Column(
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )
    result_text = ft.Text("", size=20)
    btn_next_step = ft.ElevatedButton(
        "Далі ➡️", bgcolor=ft.Colors.GREEN, color=ft.Colors.WHITE, visible=False
    )

    def next_step(e=None):
        state["step"] += 1
        render_step()

    btn_next_step.on_click = next_step

    # ==========================================
    # АНІМОВАНИЙ ПАЛЬЧИК ДЛЯ ЧИТАННЯ
    # ==========================================
    reading_finger = ft.Container(
        content=ft.Image(
            src="finger.png", width=80, height=80, fit=ft.ImageFit.CONTAIN
        ),
        left=70,
        top=100,
        opacity=0,
        animate_position=ft.animation.Animation(0, ft.AnimationCurve.LINEAR),
        animate_opacity=ft.animation.Animation(300, ft.AnimationCurve.EASE_IN_OUT),
        visible=False,
    )

    def animate_finger_sweep():
        while True:
            try:
                if state.get("reading_mode") and reading_finger.visible:
                    reading_finger.animate_position = ft.animation.Animation(
                        0, ft.AnimationCurve.LINEAR
                    )
                    reading_finger.left = 70
                    reading_finger.opacity = 0
                    reading_finger.update()
                    time.sleep(0.1)

                    if not state.get("reading_mode"):
                        continue

                    reading_finger.animate_position = ft.animation.Animation(
                        2000, ft.AnimationCurve.LINEAR
                    )
                    reading_finger.left = 270
                    reading_finger.opacity = 1
                    reading_finger.update()
                    time.sleep(2.0)

                    if not state.get("reading_mode"):
                        continue

                    reading_finger.opacity = 0
                    reading_finger.update()
                    time.sleep(0.3)
                else:
                    time.sleep(0.2)
            except Exception:
                time.sleep(0.2)

    threading.Thread(target=animate_finger_sweep, daemon=True).start()

    # ==========================================
    # РОЗУМНИЙ МІКРОФОН
    # ==========================================
    mic_icon = ft.Icon(name=ft.Icons.MIC, size=50, color=ft.Colors.WHITE)

    def button_clicked(e):
        if not state["can_record"]:
            return

        if not state["recording"]:
            audio_rec.start_recording("live_sample_u.wav")
            state["recording"] = True

            mic_button.bgcolor = ft.Colors.RED_700
            mic_icon.name = ft.Icons.STOP
            mic_button.scale = 1.0
            mic_button.shadow = ft.BoxShadow(
                spread_radius=1, blur_radius=15, color=ft.Colors.RED_200
            )

            result_text.value = "Слухаю..."
            page.update()
        else:
            audio_rec.stop_recording()
            state["recording"] = False
            state["can_record"] = False

            mic_button.bgcolor = ft.Colors.BLUE
            mic_icon.name = ft.Icons.MIC
            mic_button.shadow = ft.BoxShadow(
                spread_radius=1, blur_radius=15, color=ft.Colors.BLUE_200
            )

            result_text.value = "Перевіряю..."
            page.update()
            time.sleep(0.5)

            try:
                pred_phoneme, conf = predict_phoneme("live_sample_u.wav")
                
                # 1. Приводимо все до нижнього регістру та прибираємо пробіли
                pred_str = str(pred_phoneme).strip().lower()
                target_str = state["current_target"].strip().lower()
                
                # 2. Вирішуємо проблему "Кирилиця vs Латиниця"
                # (замінюємо українські букви на англійські для чесного порівняння)
                pred_str = pred_str.replace("у", "u").replace("а", "a")
                target_str = target_str.replace("у", "u").replace("а", "a")

                # 3. Гнучка перевірка для розтягнутих звуків (імітація немовляти/луни)
                acceptable_matches = [target_str]
                
                if state["step"] == 24:
                    acceptable_matches.extend(["la", "a", "u"])
                elif state["step"] == 25:
                    acceptable_matches.extend(["a", "u", "la"])

                if pred_str in acceptable_matches and conf >= 8.0:
                    result_text.value = f"Молодець! (Впевненість: {conf:.1f}%)"
                    result_text.color = "green"
                    kapi_speak("Молодець! Дуже гарно!", "kapi_good_reading.wav")
                    trigger_success()
                    next_step()
                else:
                    # Тепер ви бачитимете на екрані, скільки відсотків не вистачило
                    result_text.value = f"Спробуй ще раз! (Почуто: {pred_str}, впевненість: {conf:.1f}%)"
                    result_text.color = "orange"

                    # Визначаємо емоцію залежно від цільової літери
                    face_type = "u" if "u" in target_str else "a"
                    kapi_speak("Спробуй ще раз!", "kapi_try_again.wav", face="sad")

                    def revert_after_fail():
                        time.sleep(1.5)
                        kapitoshka_img.src = f"kapitoshka_{face_type}.png"
                        page.update()
                        time.sleep(2)
                        kapitoshka_img.src = "kapitoshka_idle.png"
                        state["can_record"] = True
                        page.update()

                    threading.Thread(target=revert_after_fail, daemon=True).start()

            except Exception as ex:
                result_text.value = f"Помилка: {ex}"
                result_text.color = "red"
                state["can_record"] = True
            page.update()

    mic_button = ft.Container(
        content=mic_icon,
        width=100,
        height=100,
        bgcolor=ft.Colors.BLUE,
        shape=ft.BoxShape.CIRCLE,
        alignment=ft.alignment.center,
        shadow=ft.BoxShadow(spread_radius=1, blur_radius=15, color=ft.Colors.BLUE_200),
        animate_scale=ft.animation.Animation(500, ft.AnimationCurve.EASE_IN_OUT),
        scale=1.0,
        visible=False,
        on_click=button_clicked,
    )

    def pulse_mic():
        while True:
            try:
                if (
                    mic_button.visible
                    and state["can_record"]
                    and not state["recording"]
                ):
                    mic_button.scale = 1.15 if mic_button.scale == 1.0 else 1.0
                else:
                    mic_button.scale = 1.0
                mic_button.update()
            except Exception:
                pass
            time.sleep(0.5)

    threading.Thread(target=pulse_mic, daemon=True).start()

    # ==========================================
    # ЛОГІКА МІНІ-ІГОР (Вибір картинки)
    # ==========================================
    def make_game_choice(is_correct, word, text_widget, success_audio):
        if is_correct:
            new_spans = []
            for char in word:
                # Підсвічуємо ВСІ літери "у" та "У"
                if char.lower() == "у":
                    new_spans.append(
                        ft.TextSpan(
                            char, ft.TextStyle(color=ft.Colors.RED, weight="bold")
                        )
                    )
                else:
                    new_spans.append(
                        ft.TextSpan(
                            char, ft.TextStyle(color=ft.Colors.BLACK, weight="bold")
                        )
                    )

            text_widget.spans = new_spans
            kapi_speak(f"Правильно! {word.capitalize()}", success_audio)
            trigger_success()
            time.sleep(1.0)
            next_step()
        else:
            kapi_speak(
                "Подумай ще! Обери слово, яке починається на букву У.", "kapi_wrong_u.wav", face="sad"
            )

            def revert_sadness():
                time.sleep(2)
                kapitoshka_img.src = "kapitoshka_idle.png"
                page.update()

            threading.Thread(target=revert_sadness, daemon=True).start()

    def create_game_item(image_src, word, is_correct, success_audio):
        text_widget = ft.Text(
            spans=[
                ft.TextSpan(word, ft.TextStyle(color=ft.Colors.BLACK, weight="bold"))
            ],
            size=36,
        )
        img = ft.Image(src=image_src, width=120, height=120)
        item = ft.GestureDetector(
            on_tap=lambda e: make_game_choice(
                is_correct, word, text_widget, success_audio
            ),
            content=ft.Column(
                [img, text_widget], horizontal_alignment=ft.CrossAxisAlignment.CENTER
            ),
        )
        return item

    # ==========================================
    # ГОЛОВНИЙ РЕНДЕР КРОКІВ (РІВЕНЬ У)
    # ==========================================
    def render_step():
        step = state["step"]
        level_content.controls.clear()

        btn_next_step.visible = False
        mic_button.visible = False
        result_text.value = ""

        state["can_record"] = False
        state["reading_mode"] = False
        reading_finger.visible = False
        reading_finger.animate_position = ft.animation.Animation(
            0, ft.AnimationCurve.LINEAR
        )
        reading_finger.left = 70
        reading_finger.opacity = 0

        # ЕТАП 1: Знайомство з літерою
        if step == 0:
            kapi_speak(
                "Привіт! Сьогодні ми вивчаємо букву У. Ця буква позначає голосний звук, як і літера А. Подивись, як ми складаємо губи дудочкою: У-у-у! Це велика літера У.",
                "kapi_intro_u_big.wav",
                face="u",
            )  
            level_content.controls.append(
                ft.Text("У", size=180, color=ft.Colors.RED_700, weight="bold")
            )
            btn_next_step.visible = True

        elif step == 1:
            kapi_speak(
                "А це маленька літера у. Губи теж дудочкою: у-у-у!",
                "kapi_intro_u_small.wav",
                face="u",
            )  
            level_content.controls.append(
                ft.Text("у", size=150, color=ft.Colors.RED_700, weight="bold")
            )
            btn_next_step.visible = True

        # ЕТАП 2: Ігри на пошук звука (у будь-якій позиції)
        elif step == 2:
            kapi_speak(
                "Що зображено на малюнках? Знайди слово, яке починається на літеру У!",
                "kapi_find_u.wav",
            )  
            row = ft.Row(
                [
                    create_game_item("udav.png", "Удав", True, "kapi_correct_udav.wav"),
                    create_game_item("cherepaha.png", "Черепаха", False, ""),
                    create_game_item("krislo.png", "Крісло", False, ""),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=40,
            )
            level_content.controls.append(row)

        elif step == 3:
            kapi_speak(
                "Що зображено на малюнках? Знайди слово, яке починається на літеру У!",
                "kapi_find_u.wav",
            )  
            row = ft.Row(
                [
                    create_game_item("dyvan.png", "Диван", False, ""),
                    create_game_item(
                        "umyvalnyk.png",
                        "Умивальник",
                        True,
                        "kapi_correct_umyvalnyk.wav",
                    ),
                    create_game_item("pizza.png", "Піца", False, ""),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=40,
            )
            level_content.controls.append(row)

        elif step == 4:
            kapi_speak(
                "Що зображено на малюнках? Знайди слово, яке починається на літеру У!",
                "kapi_find_u.wav",
            )  
            row = ft.Row(
                [
                    create_game_item("pomidor.png", "Помідор", False, ""),
                    create_game_item("vynograd.png", "Виноград", False, ""),
                    create_game_item(
                        "usmishka.png", "Усмішка", True, "kapi_correct_usmishka.wav"
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=40,
            )
            level_content.controls.append(row)

        # ЕТАП 3: Читання окремих літер (Кроки 5-19)
        elif 5 <= step <= 19:
            state["reading_mode"] = True
            reading_finger.visible = True

            # Формат: (Текст Капітошки, Текст на екрані, Розмір шрифту, Цільовий клас для SVM, Емоція Капітошки)
            tasks = [
                ("Прочитай велику букву.", "У", 120, "u", "u"),  # Step 5
                ("Прочитай маленьку букву.", "у", 120, "u", "u"),  # Step 6
                ("Пам'ятаєш, що ми читаємо зліва направо? Слідкуй за пальчиком!", "У-у-у-у", 80, "u", "u"),  # Step 7
                ("А тепер згадаємо попередню букву.", "а", 120, "a", "a"),  # Step 8
                ("А що це за буква? Назви її!", "У", 120, "u", "u"),  # Step 9
                ("А цю пам'ятаєш?", "А", 120, "a", "a"),  # Step 10
                ("Давай прочитаємо цю маленьку літеру.", "у", 120, "u", "u"),  # Step 11
                ("А тепер велика! Вимов гучно:", "У", 120, "u", "u"),  # Step 12
                ("Скажи, яка це буква?", "А", 120, "a", "a"),  # Step 13
                ("А цю зможеш прочитати?", "а", 120, "a", "a"),  # Step 14
                ("Увага, наступна літера!", "У", 120, "u", "u"),  # Step 15
                ("А це що за буква?", "а", 120, "a", "a"),  # Step 16
                ("Прочитай цю маленьку букву.", "у", 120, "u", "u"),  # Step 17
                ("Назви цю велику літеру.", "А", 120, "a", "a"),  # Step 18
                ("І знову ось ця літера!", "У", 120, "u", "u"),  # Step 19
            ]

            task_idx = step - 5
            current_task = tasks[task_idx]

            state["current_target"] = current_task[3]

            # Якщо у вас ще немає всіх аудіо для "Читаємо", можна використати один універсальний файл
            audio_file = f"kapi_read_step{step}_u.wav" 
            kapi_speak(current_task[0], audio_file, face=current_task[4])

            text_widget = ft.Text(
                current_task[1],
                size=current_task[2],
                color=ft.Colors.RED_700,
                weight="bold",
            )
            text_container = ft.Container(
                content=text_widget,
                alignment=ft.alignment.center,
                width=400,
                height=180,
            )
            reading_stack = ft.Stack(
                controls=[text_container, reading_finger], width=400, height=180
            )
            level_content.controls.append(reading_stack)

            mic_button.visible = True
            delay_seconds = max(2.5, len(current_task[0]) * 0.08)

            def enable_recording():
                time.sleep(delay_seconds)
                state["can_record"] = True

            threading.Thread(target=enable_recording, daemon=True).start()

        # ЕТАП 4: Читання складів (Кроки 20-25)
        elif 20 <= step <= 25:
            state["reading_mode"] = True
            reading_finger.visible = True

            syllable_tasks = [
                ("Тепер найцікавіше! З'єднуємо букви у склади. Слідкуй за пальчиком і читай зліва направо: Ау", "Ау", 120, "au", "au"), # Step 20
                ("А тепер навпаки: Уа", "Уа", 120, "ua", "ua"), # Step 21
                ("Маленькі букви", "ау", 120, "au", "au"), # Step 22
                ("І ще раз навпаки", "уа", 120, "ua", "ua"), # Step 23
                ("Згадай, як плаче немовля", "У-а-а-а", 80, "ua", "ua"), # Step 24
                ("А як гукає той, хто заблукав у лісі?", "А-у-у-у", 80, "au", "au"), # Step 25
            ]

            task_idx = step - 20
            current_task = syllable_tasks[task_idx]

            state["current_target"] = current_task[
                3
            ]  # Ці класи (au, ua) мають бути натреновані у вашій моделі
            kapi_speak(
                current_task[0], f"kapi_read_step{step}_u.wav", face=current_task[4]
            )

            text_widget = ft.Text(
                current_task[1],
                size=current_task[2],
                color=ft.Colors.RED_700,
                weight="bold",
            )
            text_container = ft.Container(
                content=text_widget,
                alignment=ft.alignment.center,
                width=400,
                height=180,
            )
            reading_stack = ft.Stack(
                controls=[text_container, reading_finger], width=400, height=180
            )
            level_content.controls.append(reading_stack)

            mic_button.visible = True
            delay_seconds = max(2.5, len(current_task[0]) * 0.08)

            def enable_recording():
                time.sleep(delay_seconds)
                state["can_record"] = True

            threading.Thread(target=enable_recording, daemon=True).start()

        # ФІНАЛ РІВНЯ
        elif step == 26:
            kapi_speak(
                "Ти справжній герой! Ми вивчили літеру Уу та навчилися читати перші склади!",
                "kapi_level_u_done.wav",
            )
            level_content.controls.append(
                ft.Text(
                    "РІВЕНЬ ПРОЙДЕНО!",
                    size=60,
                    color=ft.Colors.GREEN_700,
                    weight="bold",
                )
            )
            trigger_success()

            def go_back():
                time.sleep(4)
                page.window_close()  # Оскільки це окремий файл, просто закриваємо вікно після завершення

            threading.Thread(target=go_back).start()

        page.update()

    trainer_ui = ft.Column(
        [
            ft.Divider(),
            kapitoshka_ui,
            ft.Divider(),
            level_content,
            btn_next_step,
            mic_button,
            result_text,
        ],
        alignment=ft.MainAxisAlignment.START,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )

    page.add(
        ft.Stack(
            balloons
            + [
                ft.Container(
                    content=trainer_ui, alignment=ft.alignment.top_center, padding=20
                )
            ],
            expand=True,
        )
    )

    # Запускаємо перший крок
    render_step()


ft.app(target=main, assets_dir="assets")
