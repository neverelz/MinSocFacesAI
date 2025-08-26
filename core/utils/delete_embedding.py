#Скрипт удаляет человека из базы известных
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
from shared import KNOWN_DB_PATH

with open(KNOWN_DB_PATH, "rb") as f:
    db = pickle.load(f)

print("Пользователи в базе:")
for name in db.keys():
    print("-", name)

target = input("Кого удалить? > ")
if target in db:
    del db[target]
    with open(KNOWN_DB_PATH, "wb") as f:
        pickle.dump(db, f)
    print(f"[✓] {target} удалён.")
else:
    print("Пользователь не найден.")
