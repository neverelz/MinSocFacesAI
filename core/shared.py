#Скрипт подключает камеры, присваивает ID неизвестному и определяет известного человеа 
from pathlib import Path
from datetime import timedelta
import numpy as np
import os
import json
import pickle
import multiprocessing

_manager = None  # set by init_shared
daily_known = None
daily_unknown = None
camera_status = None

def init_shared(manager: multiprocessing.Manager):
    """Create shared dict proxies from the given manager. Call in main process only."""
    global _manager, daily_known, daily_unknown, camera_status
    _manager = manager
    daily_known = _manager.dict()
    daily_unknown = _manager.dict()
    camera_status = _manager.dict()

def attach_shared(dk, du, cs):
    """Attach already-created shared dict proxies. Call in main and child processes."""
    global daily_known, daily_unknown, camera_status
    daily_known = dk
    daily_unknown = du
    camera_status = cs

def _ensure_local_dicts():
    """Fallback to local dicts if attach/init wasn't called (single-process/testing)."""
    global daily_known, daily_unknown, camera_status
    if daily_known is None: daily_known = {}
    if daily_unknown is None: daily_unknown = {}
    if camera_status is None: camera_status = {}

# === Пути ===
BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "db"

KNOWN_DB_PATH = DB_DIR / "embeddings_db.pkl"
UNKNOWN_DB_PATH = DB_DIR / "unknown_db.pkl"
SAVE_DIR = BASE_DIR / "recognized_faces"
UNKNOWN_SAVE_DIR = BASE_DIR / "unknown_faces"
CONFIG_PATH = BASE_DIR / "config.json"

# === Пороговые значения и задержка ===
SIMILARITY_THRESHOLD_KNOWN = 0.65
SIMILARITY_THRESHOLD_UNKNOWN = 0.55
SCREENSHOT_DELAY = timedelta(seconds=10)

# === Камеры ===
CAMERA_SOURCES = [
    0,
    1,
    "rtsp://admin:hGm_37cD@10.12.19.68/Streaming/Channels/101"
]

# === Утилиты ===

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_mean_embedding(vectors):
    return np.mean(vectors, axis=0)

def find_best_unknown_match(embedding, unknown_db, threshold=0.55):
    best_score = 0
    best_match = None
    for unk_name, unk_embs in unknown_db.items():
        mean_emb = get_mean_embedding(unk_embs)
        score = cosine_similarity(embedding, mean_emb)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = unk_name
    return best_match

def get_new_unknown_id(db):
    existing = [int(k.split('_')[1]) for k in db if k.startswith("unknown_")]
    new_id = max(existing, default=0) + 1
    return f"unknown_{new_id}"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {
        "SIMILARITY_THRESHOLD_KNOWN": SIMILARITY_THRESHOLD_KNOWN,
        "SIMILARITY_THRESHOLD_UNKNOWN": SIMILARITY_THRESHOLD_UNKNOWN
    }

# === Daily stats and camera status (AI additions) ===
from datetime import datetime, timezone
import multiprocessing

def _append_to_managed_list(md, key, value):
    lst = list(md.get(key, []))
    lst.append(value)
    md[key] = lst

def register_known(name: str, photo_path: str):
    _ensure_local_dicts()
    if not name:
        return
    _append_to_managed_list(daily_known, name, photo_path)

def register_unknown(unknown_id: str, photo_path: str):
    _ensure_local_dicts()
    if not unknown_id:
        return
    _append_to_managed_list(daily_unknown, unknown_id, photo_path)

def reset_daily_stats():
    _ensure_local_dicts()
    daily_known.clear()
    daily_unknown.clear()

def mark_camera_status(cam_id: int, status: str):
    _ensure_local_dicts()
    camera_status[int(cam_id)] = {
        'status': str(status),
        'ts': datetime.now(timezone.utc).isoformat()
    }

# === AI config overrides ===
try:
    _cfg = load_config()
    # thresholds
    if 'SIMILARITY_THRESHOLD_KNOWN' in _cfg:
        SIMILARITY_THRESHOLD_KNOWN = float(_cfg['SIMILARITY_THRESHOLD_KNOWN'])
    if 'SIMILARITY_THRESHOLD_UNKNOWN' in _cfg:
        SIMILARITY_THRESHOLD_UNKNOWN = float(_cfg['SIMILARITY_THRESHOLD_UNKNOWN'])
    # unknown db path
    if 'UNKNOWN_DB_PATH' in _cfg:
        try:
            _p = _cfg['UNKNOWN_DB_PATH']
            if _p:
                UNKNOWN_DB_PATH = (BASE_DIR / _p) if not os.path.isabs(_p) else Path(_p)
        except Exception:
            pass
    # cameras mapping (dict of id->source) or list
    if 'CAMERAS' in _cfg:
        cams = _cfg['CAMERAS']
        if isinstance(cams, dict):
            CAMERA_SOURCES = list(cams.values())
        elif isinstance(cams, list):
            CAMERA_SOURCES = cams
except Exception:
    pass
