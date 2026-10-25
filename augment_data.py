import os
import librosa
import soundfile as sf
import numpy as np

# Шляхи до папок
SOURCE_DIRS = ["dataset/mother", "dataset/child"]
OUTPUT_DIR = "dataset_augmented"

# Створюємо базову папку для аугментованих даних
os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_white_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise


def augment_audio(file_path, output_folder, file_prefix):
    """
    Читає оригінальний файл і створює 4 нові версії.
    """
    audio, sr = librosa.load(file_path, sr=16000)
    base_name = os.path.basename(file_path).split(".")[0]

    # 1. Збереження оригіналу копією
    sf.write(
        os.path.join(output_folder, f"{file_prefix}_{base_name}_orig.wav"), audio, sr
    )

    # 2. Уповільнення (діти часто тягнуть звуки)
    audio_slow = librosa.effects.time_stretch(audio, rate=0.8)
    sf.write(
        os.path.join(output_folder, f"{file_prefix}_{base_name}_slow.wav"),
        audio_slow,
        sr,
    )

    # 3. Прискорення
    audio_fast = librosa.effects.time_stretch(audio, rate=1.2)
    sf.write(
        os.path.join(output_folder, f"{file_prefix}_{base_name}_fast.wav"),
        audio_fast,
        sr,
    )

    # 4. Зміна тону (робимо голос вищим - Pitch Shift)
    audio_high = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
    sf.write(
        os.path.join(output_folder, f"{file_prefix}_{base_name}_high.wav"),
        audio_high,
        sr,
    )

    # 5. Додавання шуму (імітація поганого мікрофона)
    audio_noisy = add_white_noise(audio)
    sf.write(
        os.path.join(output_folder, f"{file_prefix}_{base_name}_noise.wav"),
        audio_noisy,
        sr,
    )


print("Починаємо аугментацію даних...")

for src_dir in SOURCE_DIRS:
    if not os.path.exists(src_dir):
        print(f"Папка {src_dir} не знайдена. Пропускаю.")
        continue

    prefix = "mother" if "mother" in src_dir else "child"

    for filename in os.listdir(src_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(src_dir, filename)
            try:
                augment_audio(file_path, OUTPUT_DIR, prefix)
            except Exception as e:
                print(f"Помилка з файлом {filename}: {e}")

# Рахуємо результат
total_files = len(os.listdir(OUTPUT_DIR)) if os.path.exists(OUTPUT_DIR) else 0
print(f"Аугментацію завершено! У папці {OUTPUT_DIR} тепер {total_files} файлів.")
