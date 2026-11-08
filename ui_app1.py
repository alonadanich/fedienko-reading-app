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
# ІНІЦІАЛІЗАЦІЯ МОДЕЛЕЙ
# ==========================================
print("Завантаження моделей... Зачекайте.")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
processor = Wav2Vec2Processor.from_pretrained(
    "anton-l/wav2vec2-large-xlsr-53-ukrainian"
)
wav2vec_model = Wav2Vec2Model.from_pretrained(
    "anton-l/wav2vec2-large-xlsr-53-ukrainian"
)

svm_clf = joblib.load("fedienko_svm_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# ==========================================
# ЛОГІКА ПРОГРАМИ
# ==========================================
def predict_phoneme(audio_path):
    speech, rate = librosa.load(audio_path, sr=16000)
    speech_trimmed, _ = librosa.effects.trim(speech, top_db=20)
    if len(speech_trimmed) > 1000:
        speech = speech_trimmed
    speech = librosa.util.normalize(speech)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        features = wav2vec_model(**inputs).last_hidden_state
    features_mean = features.mean(dim=1).numpy()
    probs = svm_clf.predict_proba(features_mean)[0]
    max_index = np.argmax(probs)
    return label_encoder.inverse_transform([max_index])[0], probs[max_index] * 100


def main(page: ft.Page):
    page.title = "Тренажер читання Федієнка"
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

        if face == "a":
            kapitoshka_img.src = "kapitoshka_a.png"
        elif face == "sad":
            kapitoshka_img.src = "kapitoshka_sad.png"
        else:
            kapitoshka_img.src = "kapitoshka_idle.png"

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
        animate_position=ft.animation.Animation(
            0, ft.AnimationCurve.LINEAR
        ),  # Стартує з 0
        animate_opacity=ft.animation.Animation(300, ft.AnimationCurve.EASE_IN_OUT),
        visible=False,
    )

    def animate_finger_sweep():
        while True:
            try:
                if state.get("reading_mode") and reading_finger.visible:
                    # 1. Миттєве переміщення наліво (тривалість анімації = 0)
                    reading_finger.animate_position = ft.animation.Animation(
                        0, ft.AnimationCurve.LINEAR
                    )
                    reading_finger.left = 70
                    reading_finger.opacity = 0
                    reading_finger.update()

                    time.sleep(0.1)  # Даємо UI час намалювати його на старті

                    if not state.get("reading_mode"):
                        continue

                    # 2. Плавний рух направо (тривалість анімації = 2000 мс)
                    reading_finger.animate_position = ft.animation.Animation(
                        2000, ft.AnimationCurve.LINEAR
                    )
                    reading_finger.left = 270
                    reading_finger.opacity = 1
                    reading_finger.update()

                    time.sleep(2.0)  # Чекаємо поки доїде до кінця

                    if not state.get("reading_mode"):
                        continue

                    # 3. Ховаємо пальчик
                    reading_finger.opacity = 0
                    reading_finger.update()

                    time.sleep(0.3)  # Чекаємо поки повністю зникне
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
            audio_rec.start_recording("live_sample.wav")
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
                pred_phoneme, conf = predict_phoneme("live_sample.wav")
                if pred_phoneme == "a" and conf >= 20.0:
                    result_text.value = "Супер! Правильно!"
                    result_text.color = "green"
                    kapi_speak("Молодець! Дуже гарно!", "kapi_good_reading.wav")
                    trigger_success()
                    next_step()
                else:
                    result_text.value = "Спробуй ще раз! Відкрий ротик ширше!"
                    result_text.color = "orange"

                    kapi_speak(
                        "Спробуй ще раз! А-а-а!", "kapi_try_again.wav", face="sad"
                    )

                    def revert_after_fail():
                        time.sleep(1.5)
                        kapitoshka_img.src = "kapitoshka_a.png"
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
                if char.lower() == "а":
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
            kapi_speak(f"Правильно, молодець! {word.capitalize()}", success_audio)
            trigger_success()
            time.sleep(1.0)
            next_step()
        else:
            kapi_speak(
                "Подумай ще! Обери слово, яке починається на літеру А.",
                "kapi_wrong.wav",
                face="sad",
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
    # ГОЛОВНИЙ РЕНДЕР КРОКІВ
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

        if step == 0:
            kapi_speak(
                "Привіт! Поглянь на цю букву. Вона голосна, бо вміє дзвінко співати: А-а-а! Давай заспіваємо разом: А-а-а! Це — велика буква А. Вона висока, наче доросла! Великі букви завжди стоять на початку імен. Наприклад: Аліса, Артем, Андрійко. А ще вони відкривають кожне нове речення!",
                "kapi_intro_big.wav",
                face="a",
            )
            level_content.controls.append(
                ft.Text("А", size=180, color=ft.Colors.RED_700, weight="bold")
            )
            btn_next_step.visible = True

        elif step == 1:
            kapi_speak(
                "А це — маленька буква а. Вона читається так само, як і велика: а-а-а!",
                "kapi_intro_small.wav",
                face="a",
            )
            level_content.controls.append(
                ft.Text("а", size=150, color=ft.Colors.RED_700, weight="bold")
            )
            btn_next_step.visible = True

        elif step == 2:
            kapi_speak(
                "Розглянь малюнки! Обери слово, яке починається на літеру А!",
                "kapi_task1.wav",
            )
            row = ft.Row(
                [
                    create_game_item("velosyped.png", "Велосипед", False, ""),
                    create_game_item(
                        "avtobus.png", "Автобус", True, "kapi_correct_avtobus.wav"
                    ),
                    create_game_item("derevo.png", "Дерево", False, ""),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=40,
            )
            level_content.controls.append(row)

        elif step == 3:
            kapi_speak(
                "Розглянь малюнки! Обери слово, яке починається на літеру А!",
                "kapi_task2.wav",
            )
            row = ft.Row(
                [
                    create_game_item(
                        "agrus.png", "Аґрус", True, "kapi_correct_agrus.wav"
                    ),
                    create_game_item("budynok.png", "Будинок", False, ""),
                    create_game_item("raketa.png", "Ракета", False, ""),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=40,
            )
            level_content.controls.append(row)

        elif step == 4:
            kapi_speak(
                "Розглянь малюнки! Обери слово, яке починається на літеру А!",
                "kapi_task3.wav",
            )
            row = ft.Row(
                [
                    create_game_item("oko.png", "Око", False, ""),
                    create_game_item("sontse.png", "Сонце", False, ""),
                    create_game_item(
                        "ananas.png", "Ананас", True, "kapi_correct_ananas.wav"
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=40,
            )
            level_content.controls.append(row)

        elif step >= 5 and step <= 11:
            state["reading_mode"] = True
            reading_finger.visible = True

            texts = [
                (
                    "Час читати! Веди пальчиком по екрану зліва направо. Прочитай велику букву. Зверни увагу, як широко відкривається ротик: А-а-а!",
                    "А",
                    120,
                ),
                ("Молодець! А тепер прочитай маленьку букву.", "а", 120),
                ("Прочитай ці букви, веди пальчиком зліва направо", "А-А-А", 80),
                ("Тепер маленькі, теж зліва направо.", "а-а-а", 80),
                (
                    "Згадаємо, які звуки називаються голосними? Так, їх можна співати! Проспівай і потягни.",
                    "А-а-а",
                    80,
                ),
                ("І наостанок: прочитай", "а", 90),
                ("А тепер це", "А", 90),
            ]

            current_task = texts[step - 5]

            face_type = "a" if (step == 5 or step == 7) else "idle"
            kapi_speak(current_task[0], f"kapi_read_step{step}.wav", face=face_type)

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

        elif step == 12:
            kapi_speak(
                "Чудова робота! Літеру Аа пройдено - новий рівень розблоковано!",
                "kapi_level_done.wav",
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
                trainer_view.visible = False
                menu_view.visible = True
                page.update()

            threading.Thread(target=go_back).start()

        page.update()

    # ==========================================
    # ЗБІРКА ЕКРАНУ ТРЕНАЖЕРА
    # ==========================================
    trainer_ui = ft.Column(
        [
            ft.OutlinedButton(
                "⬅️ Назад до карти",
                on_click=lambda e: (
                    setattr(trainer_view, "visible", False),
                    setattr(menu_view, "visible", True),
                    page.update(),
                ),
            ),
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

    trainer_view = ft.Container(
        content=ft.Stack(
            balloons
            + [
                ft.Container(
                    content=trainer_ui, alignment=ft.alignment.top_center, padding=20
                )
            ]
        ),
        visible=False,
        expand=True,
    )

    # ==========================================
    # ГОЛОВНЕ МЕНЮ
    # ==========================================
    def start_level_1(e):
        state["step"] = 0
        menu_view.visible = False
        trainer_view.visible = True
        render_step()

    menu_column = ft.Column(
        [
            ft.Text(
                "Буквар Федієнка", size=36, weight="bold", color=ft.Colors.BLUE_900
            ),
            ft.ElevatedButton(
                "🔓 Рівень 1: Буква А",
                bgcolor=ft.Colors.GREEN_600,
                color=ft.Colors.WHITE,
                width=350,
                height=60,
                on_click=start_level_1,
            ),
            ft.ElevatedButton(
                "🔒 Рівень 2: Буква У",
                bgcolor=ft.Colors.GREY_400,
                color=ft.Colors.WHITE,
                width=350,
                height=60,
            ),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=20,
    )
    menu_view = ft.Container(
        content=menu_column, alignment=ft.alignment.center, visible=True, expand=True
    )

    page.add(menu_view, trainer_view)


ft.app(target=main, assets_dir="assets")
