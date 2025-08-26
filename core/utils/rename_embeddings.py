#Скрипт переносит из базы неизвестных в базу известных людей
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
from shared import KNOWN_DB_PATH, UNKNOWN_DB_PATH

with open(UNKNOWN_DB_PATH, "rb") as f:
    unknown_db = pickle.load(f)

with open(KNOWN_DB_PATH, "rb") as f:
    try:
        embeddings_db = pickle.load(f)
    except:
        embeddings_db = {}

if not unknown_db:
    print("Нет пользователей в unknown.")
    exit()

print("Найдено unknown пользователей:")
for i, (unk, embs) in enumerate(unknown_db.items()):
    print(f"{i + 1}: {unk} ({len(embs)} эмбендингов)")

choice = input("Введите номер unknown для переноса: ").strip()
try:
    idx = int(choice) - 1
    key = list(unknown_db.keys())[idx]
except:
    print("Неверный ввод.")
    exit()

name = input("Введите имя (пример: ivanov_ii): ").strip()

if name in embeddings_db:
    decision = input(f"Пользователь '{name}' уже существует. Добавить к нему? (y/n): ").strip().lower()
    if decision == 'y':
        embeddings_db[name].extend(unknown_db[key])
        print(f"[+] Добавлено к '{name}' ({len(unknown_db[key])} эмбендингов)")
    else:
        embeddings_db[name] = unknown_db[key]
        print(f"[!] Перезаписано '{name}'")
else:
    embeddings_db[name] = unknown_db[key]
    print(f"[✓] Добавлен новый пользователь '{name}' ({len(unknown_db[key])} эмбендингов)")

# Удалим из unknown
del unknown_db[key]

with open(KNOWN_DB_PATH, "wb") as f:
    pickle.dump(embeddings_db, f)
with open(UNKNOWN_DB_PATH, "wb") as f:
    pickle.dump(unknown_db, f)

print("[✅] Перенос завершён.")
