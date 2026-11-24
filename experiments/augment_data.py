import os
import librosa
import soundfile as sf
import numpy as np
import warnings

# Вимикаємо системні попередження від librosa та numba для чистоти консолі
warnings.filterwarnings("ignore")

INPUT_DIR = "dataset"
OUTPUT_DIR = "dataset_augmented"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise


print("Починаємо глибоку розумну аугментацію...")

all_wav_files = []
for root, dirs, files in os.walk(INPUT_DIR):
    for file in files:
        if file.endswith(".wav"):
            all_wav_files.append(os.path.join(root, file))

if not all_wav_files:
    print(f"Помилка: Не знайдено жодного .wav файлу у папці {INPUT_DIR}.")
else:
    print(f"Знайдено {len(all_wav_files)} оригінальних файлів. Генеруємо...")

for i, file_path in enumerate(all_wav_files):
    speaker_folder = os.path.basename(os.path.dirname(file_path))
    filename_clean = os.path.splitext(os.path.basename(file_path))[0]

    if "_" in filename_clean:
        base_name = filename_clean
    else:
        base_name = f"{speaker_folder}_{filename_clean}"

    y, sr = librosa.load(file_path, sr=16000)

    sf.write(os.path.join(OUTPUT_DIR, f"{base_name}_orig.wav"), y, sr)

    y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    sf.write(os.path.join(OUTPUT_DIR, f"{base_name}_pitchup.wav"), y_pitch_up, sr)

    y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
    sf.write(os.path.join(OUTPUT_DIR, f"{base_name}_pitchdown.wav"), y_pitch_down, sr)

    y_noisy = add_noise(y, noise_factor=0.005)
    sf.write(os.path.join(OUTPUT_DIR, f"{base_name}_noisy.wav"), y_noisy, sr)

    duration = librosa.get_duration(y=y, sr=sr)
    if duration > 0.25:  # Трохи збільшили базовий поріг
        try:
            y_fast = librosa.effects.time_stretch(y, rate=1.15)
            sf.write(os.path.join(OUTPUT_DIR, f"{base_name}_fast.wav"), y_fast, sr)

            y_slow = librosa.effects.time_stretch(y, rate=0.85)
            sf.write(os.path.join(OUTPUT_DIR, f"{base_name}_slow.wav"), y_slow, sr)
        except Exception:
            # Якщо librosa падає на специфічному файлі, просто додаємо шум
            y_very_noisy = add_noise(y, noise_factor=0.01)
            sf.write(
                os.path.join(OUTPUT_DIR, f"{base_name}_verynoisy.wav"), y_very_noisy, sr
            )
    else:
        y_very_noisy = add_noise(y, noise_factor=0.01)
        sf.write(
            os.path.join(OUTPUT_DIR, f"{base_name}_verynoisy.wav"), y_very_noisy, sr
        )

    if (i + 1) % 10 == 0:
        print(f"Оброблено {i + 1} / {len(all_wav_files)} файлів...")

print(f"\nАугментацію завершено! Перевірте папку {OUTPUT_DIR}.")
