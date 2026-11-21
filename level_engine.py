import os
import shutil
import subprocess
import sys
import threading
import time
import flet as ft
import soundfile as sf
from levels_config import CORRECT_PHRASES, LEVELS, VOWELS, WORD_IMAGES
import re
import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None
    HAS_SOUNDDEVICE = False


class DirectMicRecorder:
    """Записує звук із мікрофона напряму через Python як у Desktop, так і в локальному --web режимі."""

    def __init__(self, samplerate: int = 16000):
        self.samplerate = samplerate
        self._frames = []
        self._stream = None

    def start(self, out_path: str):
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        self._frames = []

        def _callback(indata, frames, time_info, status):
            self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            callback=_callback,
        )
        self._stream.start()

    def stop(self, out_path: str) -> bool:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if not self._frames:
            return False
        audio_data = np.concatenate(self._frames, axis=0)
        sf.write(out_path, audio_data, self.samplerate, subtype="PCM_16")
        return True
    
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

FACE_FILES = {
    "a": "kapitoshka_a.png",
    "u": "kapitoshka_u.png",
    "o": "kapitoshka_o.png",
    "m": "kapitoshka_m.png",
    "y": "kapitoshka_y.png",
    "i": "kapitoshka_i.png",
    "sad": "kapitoshka_sad.png",
    "idle": "kapitoshka_idle.png",
}


def _clean_stem(fname: str) -> str:
    name = fname.strip().lower()
    changed = True
    while changed:
        changed = False
        for ext in (".wav", ".m4a", ".mp3", ".ogg", ".aac"):
            if name.endswith(ext):
                name = name[: -len(ext)].strip()
                changed = True
    return name


def resolve_asset_wav(filename: str) -> str | None:
    """Знаходить аудіо у папці assets навіть із подвійним .wav.wav, пробілами або у форматі .m4a/.mp3."""
    if not filename or not os.path.exists(ASSETS_DIR):
        return None

    target_stem = _clean_stem(filename)
    clean_wav_name = f"{target_stem}.wav"
    clean_wav_full = os.path.join(ASSETS_DIR, clean_wav_name)

    if os.path.exists(clean_wav_full):
        return clean_wav_name

    # Шукаємо серед усіх файлів у папці assets за чистою назвою без розширень
    for existing in os.listdir(ASSETS_DIR):
        full_existing = os.path.join(ASSETS_DIR, existing)
        if not os.path.isfile(full_existing):
            continue
        if _clean_stem(existing) == target_stem:
            ext = os.path.splitext(existing)[1].lower()
            # Якщо запис з диктофона ще у .m4a/.aac/.ogg — автоматично конвертуємо у .wav
            if ext in (".m4a", ".aac", ".ogg"):
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", full_existing, "-ar", "16000", "-ac", "1", clean_wav_full],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    print(f"🔄 Автоконвертовано у WAV: {existing} -> {clean_wav_name}")
                    return clean_wav_name
                except Exception as err:
                    print(f"⚠️ Не вдалося конвертувати {existing} через FFmpeg: {err}")
                    return existing
            # Якщо файл мав подвійне розширення (.wav.wav) або пробіл — створюємо чисту копію
            if ext == ".wav" and existing != clean_wav_name:
                shutil.copy2(full_existing, clean_wav_full)
                return clean_wav_name
            return existing

    return None


def get_wav_duration(rel_wav: str | None, fallback_text: str = "") -> float:
    if rel_wav:
        full_path = os.path.join(ASSETS_DIR, rel_wav)
        try:
            return float(sf.info(full_path).duration)
        except Exception:
            pass
    return max(2.2, len(fallback_text) * 0.075)


def play_kapi_audio(kapi_voice: ft.Audio, audio_name: str, fallback_text: str = "") -> float:
    """Надійно запускає звук у Flet на Windows і логує результат у консоль."""
    rel_wav = resolve_asset_wav(audio_name)
    if not rel_wav:
        print(f"⚠️ АУДІО НЕ ЗНАЙДЕНО в assets/: '{audio_name}'")
        return max(2.2, len(fallback_text) * 0.075)

    print(f"🔊 Відтворюю: assets/{rel_wav}")
    if kapi_voice.src == rel_wav:
        try:
            kapi_voice.seek(0)
        except Exception:
            pass
        kapi_voice.play()
    else:
        kapi_voice.src = rel_wav
        kapi_voice.autoplay = True
        kapi_voice.update()

    return get_wav_duration(rel_wav, fallback_text)

def build_colored_reading_text(display_str: str, font_size: int) -> ft.Text:
    spans = []
    for ch in display_str:
        if ch in VOWELS:
            color = ft.Colors.RED_700
        elif ch in ("┆", "|"):
            color = ft.Colors.GREY_500
        elif ch.isalpha():
            color = ft.Colors.BLUE_700
        else:
            color = ft.Colors.BLACK87
        spans.append(ft.TextSpan(ch, ft.TextStyle(color=color, weight=ft.FontWeight.BOLD)))
    return ft.Text(spans=spans, size=font_size, text_align=ft.TextAlign.CENTER)

STACK_WIDTH = 760
STACK_HEIGHT = 215
FINGER_TIP_OFFSET_X = 24  # Зміщення кінчика вказівного пальця всередині картинки finger.png


def _estimate_text_px(text_str: str, font_size: int) -> float:
    narrow_chars = set("іІїЇlI1-|┆.,!? ")
    wide_chars = set("МмЖжШшЩщЮюФфИиЫ")
    px = 0.0
    for ch in text_str:
        if ch in narrow_chars:
            px += font_size * 0.33
        elif ch in wide_chars:
            px += font_size * 0.74
        else:
            px += font_size * 0.62
    return px


def prepare_tight_rebus_image(img_filename: str, target_height: int) -> tuple[str, int]:
    """Обрізає порожні поля зліва і справа у картинці-ребусі, щоб вона стояла впритул до слів і крапки."""
    full_src = os.path.join(ASSETS_DIR, img_filename)
    if not os.path.exists(full_src):
        return img_filename, target_height

    try:
        from PIL import Image

        stem, _ = os.path.splitext(img_filename)
        tight_name = f"_tight_{stem}.png"
        tight_full = os.path.join(ASSETS_DIR, tight_name)

        if not os.path.exists(tight_full):
            with Image.open(full_src) as im:
                rgba = im.convert("RGBA")
                bbox = rgba.getchannel("A").getbbox()
                if bbox == (0, 0, rgba.width, rgba.height):
                    gray = rgba.convert("L")
                    mask = gray.point(lambda p: 255 if p < 248 else 0)
                    content_bbox = mask.getbbox()
                    if content_bbox:
                        bbox = content_bbox
                if bbox:
                    rgba = rgba.crop(bbox)
                rgba.save(tight_full)

        with Image.open(tight_full) as tim:
            w, h = tim.size
            calc_w = max(24, int(round(target_height * (w / max(1, h)))))
        return tight_name, calc_w
    except Exception:
        return img_filename, target_height


def build_reading_line(display_str: str, font_size: int) -> tuple[ft.Control, float, bool]:
    """Збирає рядок читання. Якщо є {img.png}, ставить картинку впритул у речення перед крапкою."""
    if "{" not in display_str or "}" not in display_str:
        return build_colored_reading_text(display_str, font_size), _estimate_text_px(display_str, font_size), False

    parts = re.split(r"(\{[^{}]+\})", display_str)
    rebus_h = int(font_size * 1.55)
    baseline_lift = int(font_size * 0.16)  # Вирівнює низ картинки точно по лінії літер і крапки

    controls = []
    total_px = 0.0

    for idx, part in enumerate(parts):
        if not part:
            continue

        if part.startswith("{") and part.endswith("}"):
            raw_img = part[1:-1].strip()
            tight_src, img_w = prepare_tight_rebus_image(raw_img, rebus_h)
            controls.append(
                ft.Container(
                    content=ft.Image(src=tight_src, width=img_w, height=rebus_h, fit=ft.ImageFit.CONTAIN),
                    padding=ft.padding.only(bottom=baseline_lift),
                )
            )
            total_px += img_w
        else:
            has_leading_space = part[0].isspace()
            has_trailing_space = part[-1].isspace()
            clean_part = part.strip()
            if not clean_part:
                controls.append(ft.Container(width=8))
                total_px += 8
                continue

            # Мінімальний відступ після попередньої картинки (2px перед крапкою, 8px перед словом)
            if idx > 0 and parts[idx - 1].startswith("{"):
                gap_before = 8 if has_leading_space else 2
                controls.append(ft.Container(width=gap_before))
                total_px += gap_before

            controls.append(build_colored_reading_text(clean_part, font_size))
            total_px += _estimate_text_px(clean_part, font_size)

            # Щільний відступ 8px між останнім словом і картинкою
            if idx + 1 < len(parts) and parts[idx + 1].startswith("{") and has_trailing_space:
                controls.append(ft.Container(width=8))
                total_px += 8

    row = ft.Row(
        controls=controls,
        spacing=0,
        alignment=ft.MainAxisAlignment.CENTER,
        vertical_alignment=ft.CrossAxisAlignment.END,
    )
    return row, total_px, True


def compute_finger_sweep(content_px: float, font_size: int, has_inline_rebus: bool = False) -> tuple[int, int, int, int, int]:
    """Обчислює (stack_width, start_left, end_left, top, duration_ms) під точну ширину рядка з картинкою і крапкою."""
    pad = 36
    stack_width = int(min(860, max(260, content_px + pad * 2 + 80)))
    effective_w = min(stack_width - 50, content_px)

    center_x = stack_width / 2
    tip_start = center_x - (effective_w / 2) - pad
    tip_end = center_x + (effective_w / 2) + pad

    start_left = max(8, int(tip_start - FINGER_TIP_OFFSET_X))
    end_left = min(stack_width - 60, int(tip_end - FINGER_TIP_OFFSET_X))

    extra_drop = 14 if has_inline_rebus else 0
    finger_top = int(STACK_HEIGHT / 2 + font_size * 0.28 + extra_drop)
    finger_top = max(115, min(STACK_HEIGHT - 72, finger_top))

    distance = max(120, end_left - start_left)
    duration_ms = int(min(3400, max(1500, distance * 4.5)))

    return stack_width, start_left, end_left, finger_top, duration_ms
    
def mount_level_view(
    page: ft.Page,
    level_id: int,
    speech_engine,
    audio_rec: ft.AudioRecorder,
    kapi_voice: ft.Audio,
    on_exit_to_map,
    on_level_completed=None,
) -> ft.Control:
    """Будує інтерактивний екран обраного замочка (1–10) усередині головного вікна."""
    cfg = LEVELS[level_id]
    target_letter = cfg["letter"]
    default_face = cfg.get("face", "idle")

    games = cfg["games"]
    reading_tasks = cfg["reading_tasks"]
    total_steps = 2 + len(games) + len(reading_tasks)

    state = {
        "active": True,
        "step": 0,
        "recording": False,
        "can_record": False,
        "reading_mode": False,
        "busy_transition": False,
        "current_target": "",
        "alt_targets": None,
        "finger_start": 280,
        "finger_end": 435,
        "finger_top": 118,
        "finger_ms": 1600,
    }

    def close_level(e=None):
        state["active"] = False
        state["reading_mode"] = False
        on_exit_to_map()

    balloons = [
        ft.Container(
            content=ft.Image(src="balloon.png", width=80),
            left=60, top=800, opacity=0,
            animate_position=ft.animation.Animation(2800, ft.AnimationCurve.EASE_OUT),
            animate_opacity=300,
        ),
        ft.Container(
            content=ft.Image(src="balloon.png", width=110),
            left=220, top=800, opacity=0,
            animate_position=ft.animation.Animation(3200, ft.AnimationCurve.EASE_OUT),
            animate_opacity=300,
        ),
    ]

    # Автоматично підхоплює прозорий level.png (якщо є) або ваш level.jpg
    finish_img_name = "level.png" if os.path.exists(os.path.join(ASSETS_DIR, "level.png")) else "level.jpg"
    finish_bg = "#E6F2FF" if finish_img_name.endswith(".png") else ft.Colors.BLACK

    finish_overlay = ft.Container(
        content=ft.Image(
            src=finish_img_name,
            fit=ft.ImageFit.COVER,
            width=float("inf"),
            height=float("inf"),
        ),
        bgcolor=finish_bg,
        left=0,
        top=0,
        right=0,
        bottom=0,
        visible=False,
    )

    kapitoshka_img = ft.Image(src="kapitoshka_idle.png", width=140, height=140, fit=ft.ImageFit.CONTAIN)
    dialogue_text = ft.Text("", size=18, color=ft.Colors.BLACK87)
    dialogue_bubble = ft.Container(
        content=dialogue_text,
        bgcolor=ft.Colors.LIGHT_BLUE_50,
        padding=15,
        border_radius=15,
        border=ft.border.all(2, ft.Colors.LIGHT_BLUE_200),
        width=560,
    )
    kapitoshka_ui = ft.Row([kapitoshka_img, dialogue_bubble], alignment=ft.MainAxisAlignment.CENTER)

    def set_face(face_key: str):
        fname = FACE_FILES.get(face_key, "kapitoshka_idle.png")
        kapitoshka_img.src = fname if os.path.exists(os.path.join("assets", fname)) else "kapitoshka_idle.png"

    def play_wav(audio_name: str) -> float:
        rel_wav = resolve_asset_wav(audio_name)
        if rel_wav:
            if kapi_voice.src == rel_wav:
                kapi_voice.play()
            else:
                kapi_voice.src = rel_wav
                kapi_voice.update()
        return play_kapi_audio(kapi_voice, audio_name, dialogue_text.value or "")

    def kapi_speak(text: str, audio_name: str = "", face: str = "idle") -> float:
        dialogue_text.value = text
        if face in ("au", "ua"):
            first_f, second_f = ("a", "u") if face == "au" else ("u", "a")

            def _anim_blend():
                if not state["active"]:
                    return
                set_face(first_f)
                page.update()
                time.sleep(0.6)
                if not state["active"]:
                    return
                set_face(second_f)
                page.update()
                time.sleep(0.8)
                if state["active"]:
                    set_face("idle")
                    page.update()

            threading.Thread(target=_anim_blend, daemon=True).start()
        else:
            set_face(face)

        dur = play_wav(audio_name)
        page.update()
        return dur

    def celebrate_and_advance(praise_text: str, audio_name: str):
        state["busy_transition"] = True
        dur = kapi_speak(praise_text, audio_name, face=default_face)

        def _worker():
            for b in balloons:
                b.top = -150
                b.opacity = 1
            page.update()

            time.sleep(max(2.5, dur + 0.35))
            if not state["active"]:
                return

            for b in balloons:
                b.opacity = 0
                b.top = 800
            page.update()

            state["busy_transition"] = False
            state["step"] += 1
            render_step()

        threading.Thread(target=_worker, daemon=True).start()

    level_content = ft.Column(
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )
    result_text = ft.Text("", size=20)
    btn_next_step = ft.ElevatedButton(
        "Далі ➡️", bgcolor=ft.Colors.GREEN, color=ft.Colors.WHITE, visible=False
    )

    def next_step(e=None):
        if state["busy_transition"] or not state["active"]:
            return
        state["step"] += 1
        render_step()

    btn_next_step.on_click = next_step

    reading_finger = ft.Container(
        content=ft.Image(src="finger.png", width=75, height=75, fit=ft.ImageFit.CONTAIN),
        left=60, top=105, opacity=0,
        animate_position=ft.animation.Animation(0, ft.AnimationCurve.LINEAR),
        animate_opacity=ft.animation.Animation(300, ft.AnimationCurve.EASE_IN_OUT),
        visible=False,
    )

    def animate_finger_sweep():
        while state["active"]:
            try:
                if state.get("reading_mode") and reading_finger.visible:
                    start_x = state["finger_start"]
                    end_x = state["finger_end"]
                    top_y = state["finger_top"]
                    dur_ms = state["finger_ms"]

                    reading_finger.animate_position = ft.animation.Animation(0, ft.AnimationCurve.LINEAR)
                    reading_finger.left = start_x
                    reading_finger.top = top_y
                    reading_finger.opacity = 0
                    reading_finger.update()
                    time.sleep(0.12)

                    if not state["reading_mode"] or not state["active"]:
                        continue

                    reading_finger.animate_position = ft.animation.Animation(dur_ms, ft.AnimationCurve.LINEAR)
                    reading_finger.left = end_x
                    reading_finger.opacity = 1
                    reading_finger.update()
                    time.sleep(dur_ms / 1000.0)

                    if not state["reading_mode"] or not state["active"]:
                        continue

                    reading_finger.opacity = 0
                    reading_finger.update()
                    time.sleep(0.3)
                else:
                    time.sleep(0.2)
            except Exception:
                time.sleep(0.2)

    threading.Thread(target=animate_finger_sweep, daemon=True).start()

    mic_icon = ft.Icon(name=ft.Icons.MIC, size=50, color=ft.Colors.WHITE)
    sample_file = os.path.join(BASE_DIR, f"live_sample_lvl{level_id}.wav")
    direct_rec = DirectMicRecorder(samplerate=16000) if HAS_SOUNDDEVICE else None

    def on_mic_click(e):
        if not state["can_record"] or state["busy_transition"] or not state["active"]:
            return

        if page.web and direct_rec is None:
            result_text.value = "Встановіть sounddevice: pip install sounddevice"
            result_text.color = ft.Colors.RED_700
            page.update()
            return

        if not state["recording"]:
            if direct_rec is not None:
                direct_rec.start(sample_file)
            else:
                if os.path.exists(sample_file):
                    os.remove(sample_file)
                audio_rec.start_recording(sample_file)

            state["recording"] = True
            mic_button.bgcolor = ft.Colors.RED_700
            mic_icon.name = ft.Icons.STOP
            mic_button.scale = 1.0
            result_text.value = "Слухаю..."
            result_text.color = ft.Colors.BLUE_900
            page.update()
        else:
            state["recording"] = False
            state["can_record"] = False
            mic_button.bgcolor = ft.Colors.BLUE
            mic_icon.name = ft.Icons.MIC
            result_text.value = "Перевіряю..."
            page.update()

            if direct_rec is not None:
                direct_rec.stop(sample_file)
            else:
                audio_rec.stop_recording()
                time.sleep(0.45)

            try:
                is_ok, heard, conf = speech_engine.evaluate(
                    sample_file, state["current_target"], state["alt_targets"]
                )
                if is_ok:
                    result_text.value = f"Молодець! ({conf:.0f}%)"
                    result_text.color = ft.Colors.GREEN_700
                    celebrate_and_advance("Молодець, дуже гарно!", "kapi_good_reading.wav")
                else:
                    result_text.value = f"Спробуй ще раз! (Почуто: {heard})"
                    result_text.color = ft.Colors.ORANGE_800
                    retry_txt, retry_wav = cfg.get("try_again", ("Спробуй ще раз!", "kapi_try_again.wav"))
                    dur = kapi_speak(retry_txt, retry_wav, face="sad")

                    def _unlock_retry():
                        time.sleep(max(1.8, dur + 0.25))
                        if state["active"]:
                            set_face(default_face)
                            state["can_record"] = True
                            page.update()

                    threading.Thread(target=_unlock_retry, daemon=True).start()
            except Exception as ex:
                result_text.value = f"Помилка мікрофона: {ex}"
                result_text.color = ft.Colors.RED
                state["can_record"] = True
            page.update()

    mic_button = ft.Container(
        content=mic_icon,
        width=95, height=95,
        bgcolor=ft.Colors.BLUE,
        shape=ft.BoxShape.CIRCLE,
        alignment=ft.alignment.center,
        shadow=ft.BoxShadow(spread_radius=1, blur_radius=15, color=ft.Colors.BLUE_200),
        animate_scale=ft.animation.Animation(500, ft.AnimationCurve.EASE_IN_OUT),
        scale=1.0,
        visible=False,
        on_click=on_mic_click,
    )

    def pulse_mic():
        while state["active"]:
            try:
                if mic_button.visible and state["can_record"] and not state["recording"]:
                    mic_button.scale = 1.15 if mic_button.scale == 1.0 else 1.0
                else:
                    mic_button.scale = 1.0
                mic_button.update()
            except Exception:
                pass
            time.sleep(0.5)

    threading.Thread(target=pulse_mic, daemon=True).start()

    def create_word_card(word: str, is_correct: bool, success_audio: str, praise_text: str = ""):
        text_widget = ft.Text(
            spans=[ft.TextSpan(word, ft.TextStyle(color=ft.Colors.BLACK, weight=ft.FontWeight.BOLD))],
            size=30,
        )
        img_name = WORD_IMAGES.get(word, "balloon.png")
        img_src = img_name if os.path.exists(os.path.join(ASSETS_DIR, img_name)) else "balloon.png"
        img_widget = ft.Image(src=img_src, width=115, height=115, fit=ft.ImageFit.CONTAIN)

        def handle_tap(e):
            if state["busy_transition"] or not state["active"]:
                return
            if is_correct:
                hl_color = cfg["color"]
                text_widget.spans = [
                    ft.TextSpan(
                        ch,
                        ft.TextStyle(
                            color=hl_color if ch.lower() == target_letter.lower() else ft.Colors.BLACK,
                            weight=ft.FontWeight.BOLD,
                        ),
                    )
                    for ch in word
                ]
                page.update()
                spoken_text = (
                    praise_text
                    or CORRECT_PHRASES.get(success_audio)
                    or f"Правильно! {word}!"
                )
                celebrate_and_advance(spoken_text, success_audio)
            else:
                wrong_txt, wrong_wav = cfg["wrong_game"]
                dur = kapi_speak(wrong_txt, wrong_wav, face="sad")

                def _revert_face():
                    time.sleep(max(1.8, dur))
                    if state["active"]:
                        set_face("idle")
                        page.update()

                threading.Thread(target=_revert_face, daemon=True).start()

        return ft.GestureDetector(
            on_tap=handle_tap,
            content=ft.Container(
                content=ft.Column([img_widget, text_widget], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                padding=12,
                bgcolor=ft.Colors.WHITE,
                border_radius=15,
                shadow=ft.BoxShadow(spread_radius=1, blur_radius=6, color=ft.Colors.BLACK12),
            ),
        )

    def render_step():
        if not state["active"]:
            return
        step = state["step"]
        level_content.controls.clear()
        btn_next_step.visible = False
        mic_button.visible = False
        finish_overlay.visible = False
        result_text.value = ""
        state["can_record"] = False
        state["reading_mode"] = False
        reading_finger.visible = False

        # ЕТАП 1: Велика та маленька літери
        if step == 0:
            txt, wav = cfg["intro_big"]
            kapi_speak(txt, wav, face=default_face)
            display_char = "и" if target_letter == "И" else target_letter.upper()
            level_content.controls.append(
                ft.Text(display_char, size=165, color=cfg["color"], weight=ft.FontWeight.BOLD)
            )
            btn_next_step.visible = True

        elif step == 1:
            txt, wav = cfg["intro_small"]
            kapi_speak(txt, wav, face=default_face)
            row_items = [ft.Text(target_letter.lower(), size=145, color=cfg["color"], weight=ft.FontWeight.BOLD)]
            if cfg.get("intro_image") and os.path.exists(os.path.join(ASSETS_DIR, cfg["intro_image"])):
                row_items.append(ft.Image(src=cfg["intro_image"], width=130, height=130))
            for demo_word, demo_img in cfg.get("demo_words", []):
                if os.path.exists(os.path.join(ASSETS_DIR, demo_img)):
                    word_spans = [
                        ft.TextSpan(
                            ch,
                            ft.TextStyle(
                                color=cfg["color"] if ch.lower() == target_letter.lower() else ft.Colors.BLACK,
                                weight=ft.FontWeight.BOLD,
                            ),
                        )
                        for ch in demo_word
                    ]
                    row_items.append(
                        ft.Column(
                            [
                                ft.Image(src=demo_img, width=110, height=110, fit=ft.ImageFit.CONTAIN),
                                ft.Text(spans=word_spans, size=28),
                            ],
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        )
                    )
            level_content.controls.append(ft.Row(row_items, alignment=ft.MainAxisAlignment.CENTER, spacing=40))
            btn_next_step.visible = True

        # ЕТАП 2: Дидактичні ігри з малюнками
        elif 2 <= step < 2 + len(games):
            game = games[step - 2]
            kapi_speak(game["prompt"], game["audio"], face="idle")
            cards = [
                create_word_card(item[0], item[1], item[2], item[3] if len(item) > 3 else "")
                for item in game["items"]
            ]
            level_content.controls.append(ft.Row(cards, alignment=ft.MainAxisAlignment.CENTER, spacing=35))

        # ЕТАП 3: Тренажер читання (літери -> склади -> слова -> речення з ребусами)
        elif 2 + len(games) <= step < total_steps:
            task = reading_tasks[step - (2 + len(games))]
            expected_step = step
            state["reading_mode"] = True
            state["current_target"] = task["target"]
            state["alt_targets"] = task.get("alt_targets")

            disp = task["display"]
            effective_len = len(re.sub(r"\{[^{}]+\}", "   ", disp))

            if task.get("size"):
                font_sz = task["size"]
            elif effective_len <= 2:
                font_sz = 120
            elif effective_len <= 5:
                font_sz = 96
            elif effective_len <= 15:
                font_sz = 68
            elif effective_len <= 24:
                font_sz = 50
            else:
                font_sz = 36

            reading_widget, line_px, has_rebus = build_reading_line(disp, font_sz)

            # Окремі великі ілюстрації збоку показуємо лише тоді, коли картинка не вбудована в речення через {...}
            img_keys = [
                k for k in ("image", "image2")
                if task.get(k)
                and f"{{{task[k]}}}" not in disp
                and os.path.exists(os.path.join(ASSETS_DIR, task[k]))
            ]

            custom_sz = task.get("image_size")
            if isinstance(custom_sz, (tuple, list)):
                img_w, img_h = custom_sz
            elif isinstance(custom_sz, int):
                img_w, img_h = custom_sz, int(custom_sz * 0.72)
            elif len(img_keys) >= 2:
                img_w, img_h = 165, 165
            elif effective_len > 18:
                img_w, img_h = 180, 180
            else:
                img_w, img_h = 320, 215

            image_widgets = [
                ft.Image(src=task[k], width=img_w, height=img_h, fit=ft.ImageFit.CONTAIN, border_radius=12)
                for k in img_keys
            ]

            stack_w, f_start, f_end, f_top, f_ms = compute_finger_sweep(line_px, font_sz, has_rebus)
            f_top += task.get("finger_offset_y", 0)
            stack_h = max(STACK_HEIGHT, f_top + 78)

            state["finger_start"] = f_start
            state["finger_end"] = f_end
            state["finger_top"] = f_top
            state["finger_ms"] = f_ms
            reading_finger.left = f_start
            reading_finger.top = f_top
            reading_finger.visible = True

            dur = kapi_speak(task["prompt"], task["audio"], face=task.get("face", default_face))

            text_box = ft.Container(
                content=reading_widget,
                alignment=ft.alignment.center,
                width=stack_w,
                height=STACK_HEIGHT,
            )
            reading_stack = ft.Stack([text_box, reading_finger], width=stack_w, height=stack_h)

            level_content.controls.append(
                ft.Row(
                    [reading_stack] + image_widgets,
                    alignment=ft.MainAxisAlignment.CENTER,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=20,
                )
            )
            mic_button.visible = True

            def _enable_mic_after_voice():
                time.sleep(max(1.2, dur + 0.15))
                if state["active"] and state["step"] == expected_step:
                    state["can_record"] = True
                    page.update()

            threading.Thread(target=_enable_mic_after_voice, daemon=True).start()

        # ФІНАЛ РІВНЯ: Повноекранна картинка + відкриття наступного замочка
        else:
            if on_level_completed:
                on_level_completed(level_id)
            done_txt, done_wav = cfg["done"]
            dur = kapi_speak(done_txt, done_wav, face=default_face)

            finish_overlay.visible = True
            for b in balloons:
                b.top = -150
                b.opacity = 1

            def _return_after_finish():
                time.sleep(max(3.5, dur + 0.5))
                if state["active"]:
                    close_level()

            threading.Thread(target=_return_after_finish, daemon=True).start()

        page.update()

    trainer_ui = ft.Column(
        [
            ft.Row(
                [
                    ft.OutlinedButton("⬅️ Назад до мапи замочків", on_click=close_level),
                    ft.Text(f"Замочок {level_id}: Літера {target_letter}", size=20, weight=ft.FontWeight.BOLD),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                width=860,
            ),
            ft.Divider(),
            kapitoshka_ui,
            ft.Divider(),
            level_content,
            btn_next_step,
            mic_button,
            result_text,
        ],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )

    root_stack = ft.Stack(
        [
            ft.Container(content=trainer_ui, alignment=ft.alignment.top_center, padding=20),
            finish_overlay,
        ]
        + balloons,
        expand=True,
    )

    render_step()
    return root_stack


if __name__ == "__main__":
    from model_manager import ModularSpeechEngine

    lvl_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    engine = ModularSpeechEngine()

    def _standalone(page: ft.Page):
        page.window_width = 940
        page.window_height = 740
        rec = ft.AudioRecorder(audio_encoder=ft.AudioEncoder.WAV)
        voice = ft.Audio(autoplay=True)
        page.overlay.extend([rec, voice])
        view = mount_level_view(page, lvl_arg, engine, rec, voice, on_exit_to_map=lambda: page.window_close())
        page.add(view)

    ft.app(target=_standalone, assets_dir="assets")