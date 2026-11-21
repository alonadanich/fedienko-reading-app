import os
import shutil

SRC_DIR = "dataset_augmented"
DST_DIR = os.path.join("assets", "phonemes")
os.makedirs(DST_DIR, exist_ok=True)

# Пріоритет дикторів для єдиного тембру еталонних звуків
PREFERRED_SPEAKERS = ["child", "child1", "mother", "mother1"]

found = {}
if os.path.exists(SRC_DIR):
    for fname in os.listdir(SRC_DIR):
        if not fname.endswith(".wav"):
            continue
        parts = fname.replace(".wav", "").split("_")
        if len(parts) >= 3 and parts[2] == "orig":
            speaker, label = parts[0].lower(), parts[1].lower()
            if label not in found or speaker in PREFERRED_SPEAKERS[:2]:
                found[label] = os.path.join(SRC_DIR, fname)

for label, src_path in found.items():
    shutil.copy2(src_path, os.path.join(DST_DIR, f"{label}.wav"))

print(f"Скопійовано {len(found)} чистих еталонних звуків і складів у '{DST_DIR}'!")