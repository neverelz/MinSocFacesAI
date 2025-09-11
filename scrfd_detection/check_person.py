import os
import sys
import argparse
import cv2
from recognizer import FaceRecognizer


def parse_args():
    parser = argparse.ArgumentParser(description="Проверка наличия человека в базе лиц")
    parser.add_argument("image", help="Путь к изображению лица (jpg/png)")
    parser.add_argument("--threshold", type=float, default=None, help="Порог схожести (по умолчанию как в модели)")
    parser.add_argument("--assign-id", type=str, default=None, help="Присвоить указанный ID при добавлении (иначе следующий по порядку)")
    parser.add_argument("--auto-add", action="store_true", help="Автоматически добавлять незнакомого без вопросов")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"❌ Файл не найден: {args.image}")
        sys.exit(1)

    # Инициализация распознавателя
    recognizer = FaceRecognizer()
    if args.threshold is not None:
        recognizer.threshold = float(args.threshold)

    img = cv2.imread(args.image)
    if img is None or img.size == 0:
        print("❌ Не удалось прочитать изображение")
        sys.exit(1)

    name, sim = recognizer.recognize(img)

    if name != "Неизвестно":
        print(f"✅ Найден в базе. ID: {name}. Схожесть: {sim:.3f}")
        sys.exit(0)

    print(f"🟡 Человек не найден в базе. Схожесть лучшего совпадения: {sim:.3f}")

    # Добавление в базу
    if args.auto_add:
        new_id = args.assign_id if args.assign_id else recognizer.get_next_person_id()
        recognizer.add_image_to_person(new_id, img)
        print(f"➕ Добавлен новый человек с ID {new_id}")
        sys.exit(0)

    try:
        ans = input("Добавить этого человека в базу? (y/N): ").strip().lower()
    except Exception:
        ans = "n"
    if ans not in ("y", "yes", "д", "да"):
        print("🚫 Не добавлен. Отмечен как чужой.")
        sys.exit(0)

    try:
        provided = args.assign_id or input("Введите ID (пусто для нового): ").strip()
    except Exception:
        provided = ""

    if provided:
        new_id = provided
    else:
        new_id = recognizer.get_next_person_id()

    recognizer.add_image_to_person(new_id, img)
    print(f"✅ Добавлен в базу с ID {new_id}")


if __name__ == "__main__":
    main()
