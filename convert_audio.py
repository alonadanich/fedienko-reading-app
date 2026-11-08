import os
import glob

# Шукаємо всі файли .m4a у папці raw_audio
input_files = glob.glob("raw_audio/*.m4a")

print(f"Знайдено файлів для конвертації: {len(input_files)}")

# Створюємо папку assets, якщо її раптом немає
os.makedirs("assets", exist_ok=True)

for file in input_files:
    base_name = os.path.basename(file)
    name_without_ext = os.path.splitext(base_name)[0]
    output_path = f"assets/{name_without_ext}.wav"

    # Пряма команда FFmpeg:
    # -i (вхідний файл)
    # -ar 16000 (частота 16 кГц)
    # -ac 1 (1 канал / Моно)
    # -y (перезаписати, якщо файл вже існує)
    command = f'ffmpeg -i "{file}" -ar 16000 -ac 1 "{output_path}" -y -loglevel error'

    print(f"Обробка {base_name}...")
    result = os.system(command)

    if result == 0:
        print(f"✅ Успішно: {output_path}")
    else:
        print(f"❌ Помилка конвертації: {base_name}")

print("Всі операції завершено!")
