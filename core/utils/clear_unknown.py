#Скрипт очищает базу unknown.pkl + папку unknowns
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shutil
import pickle
from pathlib import Path
from shared import UNKNOWN_SAVE_DIR, UNKNOWN_DB_PATH

# Очистка папки unknown_faces
if os.path.exists(UNKNOWN_SAVE_DIR):
    for item in os.listdir(UNKNOWN_SAVE_DIR):
        item_path = os.path.join(UNKNOWN_SAVE_DIR, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        elif os.path.isfile(item_path):
            os.remove(item_path)
    print(f"[✓] Папка '{UNKNOWN_SAVE_DIR}' очищена.")
else:
    print(f"[!] Папка '{UNKNOWN_SAVE_DIR}' не существует, создаю...")
    os.makedirs(UNKNOWN_SAVE_DIR)

# Очистка базы unknown
with open(UNKNOWN_DB_PATH, "wb") as f:
    pickle.dump({}, f)
print(f"[✓] Файл '{UNKNOWN_DB_PATH}' очищен.")

print("[✅] Все данные неизвестных удалены.")
