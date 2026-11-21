import json
import os
import threading
import time
import flet as ft
from levels_config import LEVELS, WELCOME_SPEECH, LOCKED_SPEECH
from level_engine import ASSETS_DIR, mount_level_view, resolve_asset_wav, play_kapi_audio
from model_manager import ModularSpeechEngine, load_config, save_config
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

PROGRESS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "progress.json")

# Завантажуємо важкі моделі ДО відкриття вікна Flet, щоб не блокувати аудіоплеєр на старті
print("Ініціалізація трирівневого ML-ядра... Зачекайте кілька секунд.")
SPEECH_ENGINE = ModularSpeechEngine()


def load_unlocked_set() -> set[int]:
    unlocked = {lid for lid, data in LEVELS.items() if data.get("is_unlocked", False)}
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                unlocked.update(int(x) for x in json.load(f))
        except Exception:
            pass
    return unlocked


def save_unlocked_set(unlocked: set[int]):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(unlocked), f)


def main(page: ft.Page):
    page.title = "Вчимося читати з Капітошкою"
    page.bgcolor = "#E6F2FF"
    page.padding = 0
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window.width = 960
    page.window.height = 760

    unlocked_levels = load_unlocked_set()

    # Одразу передаємо початковий src у ft.Audio, щоб плеєр Windows ініціалізувався з першої секунди
    welcome_rel = resolve_asset_wav(WELCOME_SPEECH[1])
    init_src = welcome_rel or resolve_asset_wav("kapi_good_reading.wav")
    if welcome_rel:
        print(f"🔊 Стартове привітання знайдено: assets/{welcome_rel}")
    else:
        print(f"⚠️ Стартовий файл '{WELCOME_SPEECH[1]}' не знайдено у папці {ASSETS_DIR}")

    audio_rec = ft.AudioRecorder(audio_encoder=ft.AudioEncoder.WAV)
    kapi_voice = ft.Audio(src=init_src, autoplay=bool(welcome_rel))
    page.overlay.extend([audio_rec, kapi_voice])

    root_container = ft.Container(expand=True)
    current_menu_audio = {"text": WELCOME_SPEECH[0], "wav": WELCOME_SPEECH[1]}

    # ==========================================
    # КАПІТОШКА НА ГОЛОВНОМУ ЕКРАНІ МАПИ
    # ==========================================
    menu_kapi_img = ft.Image(src="kapitoshka_idle.png", width=130, height=130, fit=ft.ImageFit.CONTAIN)
    menu_bubble_text = ft.Text(WELCOME_SPEECH[0], size=17, color=ft.Colors.BLACK87, expand=True)

    def replay_menu_speech(e=None):
        play_kapi_audio(kapi_voice, current_menu_audio["wav"], current_menu_audio["text"])

    menu_bubble = ft.Container(
        content=ft.Row(
            [
                menu_bubble_text,
                ft.IconButton(
                    icon=ft.Icons.VOLUME_UP,
                    icon_color=ft.Colors.BLUE_700,
                    tooltip="Прослухати ще раз",
                    on_click=replay_menu_speech,
                ),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        ),
        bgcolor=ft.Colors.WHITE,
        padding=15,
        border_radius=15,
        border=ft.border.all(2, ft.Colors.LIGHT_BLUE_300),
        width=600,
    )

    def menu_kapi_say(text: str, wav_name: str, face: str = "idle"):
        current_menu_audio["text"] = text
        current_menu_audio["wav"] = wav_name
        menu_bubble_text.value = text
        face_file = "kapitoshka_sad.png" if face == "sad" else "kapitoshka_idle.png"
        menu_kapi_img.src = face_file if os.path.exists(os.path.join(ASSETS_DIR, face_file)) else "kapitoshka_idle.png"
        play_kapi_audio(kapi_voice, wav_name, text)
        page.update()

        if face == "sad":
            def _back_to_smile():
                time.sleep(3.2)
                menu_kapi_img.src = "kapitoshka_idle.png"
                page.update()

            threading.Thread(target=_back_to_smile, daemon=True).start()

    locks_row_1 = ft.Row(alignment=ft.MainAxisAlignment.CENTER, spacing=20)
    locks_row_2 = ft.Row(alignment=ft.MainAxisAlignment.CENTER, spacing=20)

    def show_map_view(play_greeting: bool = False):
        rebuild_locks_grid()
        root_container.content = main_tabs
        page.update()
        if play_greeting:
            menu_kapi_say(WELCOME_SPEECH[0], WELCOME_SPEECH[1], face="idle")

    def mark_level_done(completed_id: int):
        next_id = completed_id + 1
        if next_id in LEVELS:
            unlocked_levels.add(next_id)
            save_unlocked_set(unlocked_levels)

    def on_lock_clicked(level_id: int):
        if level_id not in unlocked_levels:
            menu_kapi_say(LOCKED_SPEECH[0], LOCKED_SPEECH[1], face="sad")
            return

        level_view = mount_level_view(
            page=page,
            level_id=level_id,
            speech_engine=SPEECH_ENGINE,
            audio_rec=audio_rec,
            kapi_voice=kapi_voice,
            on_exit_to_map=lambda: show_map_view(play_greeting=False),
            on_level_completed=mark_level_done,
        )
        root_container.content = level_view
        page.update()

    def rebuild_locks_grid():
        locks_row_1.controls.clear()
        locks_row_2.controls.clear()
        for level_id, data in LEVELS.items():
            is_open = level_id in unlocked_levels
            tile = ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            data["letter"],
                            size=42,
                            weight=ft.FontWeight.BOLD,
                            color=data["color"] if is_open else ft.Colors.GREY_400,
                        ),
                        ft.Icon(
                            name=ft.Icons.LOCK_OPEN if is_open else ft.Icons.LOCK,
                            color=ft.Colors.GREEN if is_open else ft.Colors.GREY_400,
                            size=28,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                width=108,
                height=108,
                bgcolor=ft.Colors.WHITE if is_open else ft.Colors.GREY_100,
                border_radius=16,
                shadow=ft.BoxShadow(spread_radius=1, blur_radius=5, color=ft.Colors.BLACK12),
                on_click=lambda e, lid=level_id: on_lock_clicked(lid),
                ink=True,
            )
            (locks_row_1 if level_id <= 5 else locks_row_2).controls.append(tile)

    map_tab = ft.Column(
        [
            ft.Container(height=12),
            ft.Row([menu_kapi_img, menu_bubble], alignment=ft.MainAxisAlignment.CENTER, spacing=15),
            ft.Container(height=15),
            ft.Text("Обери відкритий замочок!", size=30, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_800),
            ft.Container(height=10),
            locks_row_1,
            ft.Container(height=15),
            locks_row_2,
        ],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )

    # ==========================================
    # ВКЛАДКА 2: НЕЗАЛЕЖНЕ ПІДКЛЮЧЕННЯ 3 МОДЕЛЕЙ
    # ==========================================
    cfg = load_config()
    active_target_field = {"ref": None}

    def on_pick_result(e: ft.FilePickerResultEvent):
        if active_target_field["ref"] is not None:
            if e.files:
                active_target_field["ref"].value = e.files[0].path
            elif e.path:
                active_target_field["ref"].value = e.path
            page.update()

    picker = ft.FilePicker(on_result=on_pick_result)
    page.overlay.append(picker)

    def pick_file_for(field_ctrl: ft.TextField):
        active_target_field["ref"] = field_ctrl
        picker.pick_files(allow_multiple=False)

    def pick_dir_for(field_ctrl: ft.TextField):
        active_target_field["ref"] = field_ctrl
        picker.get_directory_path()

    base_ext_field = ft.TextField(label="Базовий екстрактор Wav2Vec 2.0", value=cfg["base_extractor_id"], width=680)

    l_type = ft.Dropdown(
        label="Тип (Літери)", width=150, value=cfg["letters"]["type"],
        options=[ft.dropdown.Option("sklearn", "sklearn .pkl"), ft.dropdown.Option("hf_seq_clf", "HF Folder"), ft.dropdown.Option("keras", "Keras .keras")],
    )
    l_layer = ft.Dropdown(
        label="Шар", width=100, value=str(cfg["letters"]["layer"]),
        options=[ft.dropdown.Option(str(x), f"L{x}") for x in (8, 12, 16, 18, 24)],
    )
    l_model = ft.TextField(label="Модель літер (.pkl / .keras / HF)", value=cfg["letters"]["model_path"], width=320)
    l_scaler = ft.TextField(label="Scaler літер (необов'язково)", value=cfg["letters"]["scaler_path"], width=260)
    l_enc = ft.TextField(label="Encoder літер (.pkl)", value=cfg["letters"]["encoder_path"], width=260)

    s_type = ft.Dropdown(
        label="Тип (Склади)", width=150, value=cfg["syllables"]["type"],
        options=[ft.dropdown.Option("sklearn", "sklearn .pkl"), ft.dropdown.Option("hf_seq_clf", "HF Folder"), ft.dropdown.Option("keras", "Keras .keras")],
    )
    s_layer = ft.Dropdown(
        label="Шар", width=100, value=str(cfg["syllables"]["layer"]),
        options=[ft.dropdown.Option(str(x), f"L{x}") for x in (8, 12, 16, 18, 24)],
    )
    s_model = ft.TextField(label="Модель складів (.pkl / .keras / HF)", value=cfg["syllables"]["model_path"], width=320)
    s_scaler = ft.TextField(label="Scaler складів (необов'язково)", value=cfg["syllables"]["scaler_path"], width=260)
    s_enc = ft.TextField(label="Encoder складів (.pkl)", value=cfg["syllables"]["encoder_path"], width=260)

    w_model = ft.TextField(label="CTC-модель слів та речень (HF ID або локальна папка)", value=cfg["words"]["model_path"], width=540)
    status_models = ft.Text("", size=15, weight=ft.FontWeight.BOLD)

    def apply_models_click(e):
        new_cfg = load_config()
        new_cfg["base_extractor_id"] = base_ext_field.value.strip()
        new_cfg["letters"].update({
            "type": l_type.value, "layer": int(l_layer.value),
            "model_path": l_model.value.strip(), "scaler_path": l_scaler.value.strip(), "encoder_path": l_enc.value.strip(),
        })
        new_cfg["syllables"].update({
            "type": s_type.value, "layer": int(s_layer.value),
            "model_path": s_model.value.strip(), "scaler_path": s_scaler.value.strip(), "encoder_path": s_enc.value.strip(),
        })
        new_cfg["words"]["model_path"] = w_model.value.strip()
        save_config(new_cfg)

        status_models.value = "⏳ Перезавантаження обраних моделей..."
        status_models.color = ft.Colors.BLUE_800
        page.update()
        try:
            _, msg, sec = SPEECH_ENGINE.reload_models()
            status_models.value = f"✅ {msg} (час завантаження: {sec:.2f} с)"
            status_models.color = ft.Colors.GREEN_800
        except Exception as ex:
            status_models.value = f"❌ Помилка підключення: {ex}"
            status_models.color = ft.Colors.RED_700
        page.update()

    def unlock_all_demo(e):
        unlocked_levels.update(LEVELS.keys())
        save_unlocked_set(unlocked_levels)
        rebuild_locks_grid()
        status_models.value = "🔓 Усі 10 замочків відкрито для демонстрації!"
        status_models.color = ft.Colors.GREEN_800
        page.update()

    def reset_locks_demo(e):
        unlocked_levels.clear()
        unlocked_levels.update(lid for lid, d in LEVELS.items() if d.get("is_unlocked", False))
        save_unlocked_set(unlocked_levels)
        rebuild_locks_grid()
        status_models.value = "🔒 Прогрес скинуто до початкових відкритих замочків."
        status_models.color = ft.Colors.BLUE_800
        page.update()

    settings_tab = ft.Container(
        padding=15,
        content=ft.Column(
            [
                base_ext_field,
                ft.Divider(),
                ft.Text("1. Модель для окремих літер (1 звук — усі замочки)", weight=ft.FontWeight.BOLD),
                ft.Row([l_type, l_layer, l_model, ft.ElevatedButton("📂", on_click=lambda _: pick_file_for(l_model)), ft.ElevatedButton("📁", on_click=lambda _: pick_dir_for(l_model))], alignment=ft.MainAxisAlignment.CENTER),
                ft.Row([l_scaler, ft.ElevatedButton("📂 Scaler", on_click=lambda _: pick_file_for(l_scaler)), l_enc, ft.ElevatedButton("📂 Encoder", on_click=lambda _: pick_file_for(l_enc))], alignment=ft.MainAxisAlignment.CENTER),
                ft.Divider(),
                ft.Text("2. Модель для складів (2 звуки — усі замочки)", weight=ft.FontWeight.BOLD),
                ft.Row([s_type, s_layer, s_model, ft.ElevatedButton("📂", on_click=lambda _: pick_file_for(s_model)), ft.ElevatedButton("📁", on_click=lambda _: pick_dir_for(s_model))], alignment=ft.MainAxisAlignment.CENTER),
                ft.Row([s_scaler, ft.ElevatedButton("📂 Scaler", on_click=lambda _: pick_file_for(s_scaler)), s_enc, ft.ElevatedButton("📂 Encoder", on_click=lambda _: pick_file_for(s_enc))], alignment=ft.MainAxisAlignment.CENTER),
                ft.Divider(),
                ft.Text("3. Модель для слів та речень (3+ літери — усі замочки)", weight=ft.FontWeight.BOLD),
                ft.Row([w_model, ft.ElevatedButton("📁 Папка CTC", on_click=lambda _: pick_dir_for(w_model))], alignment=ft.MainAxisAlignment.CENTER),
                ft.Row(
                    [
                        ft.ElevatedButton("💾 Застосувати моделі", bgcolor=ft.Colors.BLUE_700, color=ft.Colors.WHITE, on_click=apply_models_click),
                        ft.OutlinedButton("🔓 Відкрити всі 10 замочків", on_click=unlock_all_demo),
                        ft.OutlinedButton("🔒 Скинути замочки", on_click=reset_locks_demo),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=15,
                ),
                status_models,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            scroll=ft.ScrollMode.AUTO,
            spacing=8,
        ),
    )

    main_tabs = ft.Tabs(
        selected_index=0,
        tabs=[
            ft.Tab(text="🗺️ Країна літер (10 замочків)", content=map_tab),
            ft.Tab(text="⚙️ Керування 3 моделями та доступом", content=settings_tab),
        ],
        expand=True,
    )

    page.add(root_container)
    show_map_view(play_greeting=False)


ft.app(target=main, assets_dir=ASSETS_DIR)