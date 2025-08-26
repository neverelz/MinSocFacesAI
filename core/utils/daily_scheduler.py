import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import os
import pandas as pd
import json
from utils.telegram_alert import send_daily_report
from shared import load_config

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVENTS_DIR = os.path.join(BASE_DIR, "logs", "events")
REPORTS_DIR = os.path.join(BASE_DIR, "logs", "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_daily_report():
    now = datetime.now()
    if now.hour < 14:
        return

    date_str = now.strftime("%Y-%m-%d")
    csv_file = os.path.join(EVENTS_DIR, f"{date_str}.csv")
    report_file = os.path.join(REPORTS_DIR, f"{date_str}.json")

    if not os.path.exists(csv_file):
        return

    try:
        df = pd.read_csv(csv_file)
        known = df[df["name"].str.startswith("unknown_") == False]
        unknown = df[df["name"].str.startswith("unknown_")]

        report = {
            "date": date_str,
            "known": known.to_dict(orient="records"),
            "unknown_count": int(unknown["name"].nunique())
        }

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        config = load_config()
        token = config.get("TELEGRAM_TOKEN")
        chat_id = config.get("TELEGRAM_CHAT_ID")
        if token and chat_id:
            send_daily_report(report_file, token, chat_id)

    except Exception as e:
        print("[ERROR] Не удалось создать отчёт:", str(e))

def start_scheduler():
    scheduler = BackgroundScheduler(timezone="Europe/Moscow")
    scheduler.add_job(generate_daily_report, "cron", hour=14, minute=0)
    scheduler.start()

# === AI additions ===
from shared import daily_known, daily_unknown, reset_daily_stats
from utils.telegram_alert import send_daily_report_AI

def _generate_daily_report_from_shared():
    try:
        from datetime import datetime
        date_str = datetime.now().strftime('%Y-%m-%d')
        known = {k: list(v) for k, v in dict(daily_known).items()}
        unknown = {k: list(v) for k, v in dict(daily_unknown).items()}
        send_daily_report_AI(date_str, known, unknown)
        reset_daily_stats()
        print('[DailyScheduler AI] Report sent and stats reset.')
    except Exception as e:
        print('[DailyScheduler AI ERROR]', e)


# === Config-driven schedule (AI patch) ===
from shared import load_config
def _get_schedule_from_config():
    cfg = load_config()
    tz = cfg.get('TIMEZONE', 'Europe/Moscow')
    t = cfg.get('DAILY_REPORT_TIME', '18:00')
    try:
        hh, mm = map(int, t.split(':'))
    except Exception:
        hh, mm = 14, 0
    return tz, hh, mm

def start_async_AI():
    scheduler = BackgroundScheduler(timezone=_get_schedule_from_config()[0])
    _, hour, minute = _get_schedule_from_config()
    scheduler.add_job(_generate_daily_report_from_shared, 'cron', hour=hour, minute=minute)
    scheduler.start()
    print(f"[DailyScheduler AI] Started cron job at {hour:02d}:{minute:02d} {_get_schedule_from_config()[0]}")
