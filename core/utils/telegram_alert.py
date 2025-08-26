import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
import cv2
import requests
import json
from utils.logger import log_error
from shared import load_config

config = load_config()
TOKEN = config.get("TELEGRAM_TOKEN")
CHAT_IDS = config.get("TELEGRAM_CHAT_ID", [])

def send_telegram_alert(image, label="Неизвестный"):
    try:
        _, img_encoded = cv2.imencode('.jpg', image)
        files = {'photo': ("unknown.jpg", img_encoded.tobytes(), 'image/jpeg')}

        for chat_id in CHAT_IDS:
            data = {"chat_id": chat_id, "caption": f"⚠ Обнаружен {label}"}
            response = requests.post(
                f"https://api.telegram.org/bot{TOKEN}/sendPhoto",
                data=data,
                files=files
            )
            print(f"[Telegram DEBUG] {chat_id} → {response.status_code}: {response.text}")

    except Exception as e:
        log_error(f"[Telegram ERROR] {e}")

def send_daily_report(file_path: str, token: str = TOKEN, chat_ids: list = CHAT_IDS):
    try:
        for chat_id in chat_ids:
            with open(file_path, "rb") as f:
                files = {'document': f}
                data = {'chat_id': chat_id, 'caption': '📊 Ежедневный отчёт'}
                response = requests.post(
                    f"https://api.telegram.org/bot{token}/sendDocument",
                    data=data,
                    files=files
                )
                print(f"[Telegram DEBUG] {chat_id} → {response.status_code}: {response.text}")

    except Exception as e:
        log_error(f"[Telegram ERROR] {e}")



# === AI additions ===
def send_media_group_AI(items, token: str = None, chat_ids=None):
    # Send up to 10 photos as a single media group to each chat.
    # items: list of {'photo': filepath, 'caption': optional str}
    try:
        cfg = load_config()
        token = token or cfg.get('TELEGRAM_TOKEN')
        chat_ids = chat_ids or cfg.get('TELEGRAM_CHAT_ID', [])
        if isinstance(chat_ids, str):
            chat_ids = [chat_ids]
        if not token or not chat_ids:
            return

        batch = items[:10]
        files = {}
        media = []
        for idx, it in enumerate(batch):
            key = f'file{idx}'
            files[key] = open(it['photo'], 'rb')
            media.append({
                'type': 'photo',
                'media': f'attach://{key}',
                'caption': it.get('caption', '')
            })

        for chat_id in chat_ids:
            data = {'chat_id': chat_id, 'media': json.dumps(media, ensure_ascii=False)}
            resp = requests.post(f'https://api.telegram.org/bot{token}/sendMediaGroup', data=data, files=files)
            print(f'[Telegram media group] {chat_id}: {resp.status_code} {resp.text}')

        for f in files.values():
            try: f.close()
            except: pass

    except Exception as e:
        log_error(f'[Telegram ERROR media group] {e}')

def send_daily_report_AI(date_str, known: dict, unknown: dict, token: str = None, chat_ids=None):
    # Send textual summary for the day + one photo per unknown (if exists).
    try:
        cfg = load_config()
        token = token or cfg.get('TELEGRAM_TOKEN')
        chat_ids = chat_ids or cfg.get('TELEGRAM_CHAT_ID', [])
        if isinstance(chat_ids, str):
            chat_ids = [chat_ids]
        if not token or not chat_ids:
            return

        names = sorted(known.keys())
        unknown_count = len(unknown)
        lines = [f'📊 Дневной отчёт за {date_str}',
                 f'✅ Известные ({len(names)}): ' + (', '.join(names) if names else '—'),
                 f'❓ Неизвестных уникальных: {unknown_count}']
        text = '\n'.join(lines)

        for chat_id in chat_ids:
            r = requests.post(f'https://api.telegram.org/bot{token}/sendMessage', data={'chat_id': chat_id, 'text': text})
            print(f'[Telegram daily text] {chat_id}: {r.status_code}')
            items = []
            for uid, photos in unknown.items():
                if photos:
                    items.append({'photo': photos[0], 'caption': uid})
                if len(items) == 10:
                    send_media_group_AI(items, token=token, chat_ids=[chat_id])
                    items = []
            if items:
                send_media_group_AI(items, token=token, chat_ids=[chat_id])

    except Exception as e:
        log_error(f'[Telegram ERROR daily report AI] {e}')
