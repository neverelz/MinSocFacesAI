import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import os
import csv
import logging
from datetime import datetime

LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logs"))
ERRORS_DIR = os.path.join(LOGS_DIR, "errors")
EVENTS_DIR = os.path.join(LOGS_DIR, "events")

os.makedirs(ERRORS_DIR, exist_ok=True)
os.makedirs(EVENTS_DIR, exist_ok=True)

def log_error(message: str):
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(ERRORS_DIR, f"{today}.log")
    logging.basicConfig(filename=log_file, level=logging.ERROR, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.error(message)

def log_event(name: str):
    today = datetime.now().strftime("%Y-%m-%d")
    csv_file = os.path.join(EVENTS_DIR, f"{today}.csv")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_header = not os.path.exists(csv_file)
    with open(csv_file, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["time", "name"])
        writer.writerow([now_str, name])
