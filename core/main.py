import multiprocessing as mp
#Основной скрипт, инициализирует веб сервер Flask, подключает все остальные скрипты для работы программы
from flask import Flask, render_template, Response, jsonify, request
import os
import pickle
import shutil
import json
from shared import UNKNOWN_DB_PATH, KNOWN_DB_PATH, BASE_DIR, load_config, CAMERA_SOURCES, attach_shared
from utils.daily_scheduler import start_scheduler, start_async_AI
from pathlib import Path

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static")
)

global_frame_store = None
global_processes = []
global_stop_flags = []
frame_queue = None
result_queue = None

@app.route('/')
def index():
    cfg = load_config()
    cam_keys = list(cfg.get('CAMERAS', {}).keys()) if isinstance(cfg.get('CAMERAS'), dict) else []
    return render_template('index.html', num_cameras=len(cam_keys) or 3, camera_keys=cam_keys)

@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    def gen():
        import time
        import cv2
        while True:
            if global_frame_store is None:
                time.sleep(0.5)
                continue
            frame = global_frame_store.get(cam_id)
            if frame is None:
                time.sleep(0.1)
                continue
            ret, jpeg = cv2.imencode('.jpg', frame)
            yield b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_all', methods=['POST'])
def start_all():
    global global_processes, global_stop_flags
    global global_frame_store, frame_queue, result_queue

    if global_processes:
        return jsonify({'status': 'Камеры уже запущены'})

    from multiprocessing import Manager
    manager = Manager()
    frame_queue = manager.Queue(maxsize=10)
    result_queue = manager.Queue(maxsize=10)
    global_frame_store = manager.dict()

    from face_worker import launch_all
    global_processes, global_stop_flags = launch_all(global_frame_store, frame_queue, result_queue)
    return jsonify({'status': 'Камеры запущены'})

@app.route('/stop_all', methods=['POST'])
def stop_all():
    global global_processes, global_stop_flags, frame_queue, result_queue, global_frame_store

    for flag in global_stop_flags:
        flag.value = True
    for proc in global_processes:
        proc.join()

    global_processes.clear()
    global_stop_flags.clear()
    frame_queue = None
    result_queue = None
    global_frame_store = None

    return jsonify({'status': 'Камеры остановлены'})

@app.route('/clear_unknown', methods=['POST'])
def clear_unknown():
    os.system("start cmd /K python core/utils/clear_unknown.py")
    return jsonify({'status': 'Unknown очищены'})

@app.route('/known_faces', methods=['GET', 'POST'])
def known_faces():
    names = []
    message = None
    selected = request.args.get("person")
    if os.path.exists(KNOWN_DB_PATH):
        with open(KNOWN_DB_PATH, "rb") as f:
            db = pickle.load(f)
            names = sorted(db.keys())
    else:
        db = {}

    if request.method == 'POST':
        name = request.form.get("delete_name")
        if name and name in db:
            del db[name]
            with open(KNOWN_DB_PATH, "wb") as f:
                pickle.dump(db, f)
            folder = BASE_DIR / "recognized_faces" / name
            if os.path.exists(folder):
                shutil.rmtree(folder)
            names.remove(name)
            selected = None
            message = f"Пользователь '{name}' удалён."

    return render_template("known_faces.html", names=names, selected=selected, message=message)

@app.route('/open_folder', methods=['GET'])
def open_known():
    folder = BASE_DIR / "recognized_faces"
    if not os.path.exists(folder):
        os.makedirs(folder)
    os.system(f'start explorer "{folder}"')
    return jsonify({"status": "Папка открыта"})

@app.route('/rename_unknown', methods=['GET'])
def rename_unknown():
    os.system("start cmd /K python core/utils/rename_embeddings.py")
    return "<h2>Окно для переноса unknown открыто. Закройте его после завершения.</h2>"

@app.route('/config', methods=['GET', 'POST'])
def update_config():
    config_path = BASE_DIR / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    if request.method == 'POST':
        new_known = float(request.form.get("known_threshold", config["SIMILARITY_THRESHOLD_KNOWN"]))
        new_unknown = float(request.form.get("unknown_threshold", config["SIMILARITY_THRESHOLD_UNKNOWN"]))
        config["SIMILARITY_THRESHOLD_KNOWN"] = new_known
        config["SIMILARITY_THRESHOLD_UNKNOWN"] = new_unknown
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        return render_template("config.html", config=config, message="Порог обновлён!")

    return render_template("config.html", config=config)

def run_web(latest_frames, frame_q, result_q):
    global global_frame_store, frame_queue, result_queue
    global_frame_store = latest_frames
    frame_queue = frame_q
    result_queue = result_q
    start_scheduler()
    start_async_AI()
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    mp.freeze_support()
    manager = mp.Manager()
    dk = manager.dict(); du = manager.dict(); cs = manager.dict()
    attach_shared(dk, du, cs)
    from utils.daily_scheduler import start_async_AI
    from multiprocessing import Manager
    manager = Manager()
    frame_q = manager.Queue(maxsize=10)
    result_q = manager.Queue(maxsize=10)
    latest_frames = manager.dict()
    run_web(latest_frames, frame_q, result_q)

@app.route("/daily_report")
def daily_report():
    date_str = datetime.now().strftime("%Y-%m-%d")
    report_path = BASE_DIR / "logs" / "reports" / f"{date_str}.json"
    if not report_path.exists():
        return "<h2>Отчёт ещё не сформирован. Дождитесь 18:00 по МСК.</h2>"
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return render_template("report.html", date=data["date"], known=data["known"], unknown_count=data["unknown_count"], message=None)


@app.route('/group_unknown', methods=['POST'])
def group_unknown_route():
    try:
        from utils.unknown_grouper import group_unknown
        import pickle
        config = load_config()
        threshold = float(request.args.get('threshold', config.get('SIMILARITY_THRESHOLD_UNKNOWN', 0.85)))
        db_path = config.get('UNKNOWN_DB_PATH', str(UNKNOWN_DB_PATH))
        with open(db_path, 'rb') as f:
            unknown_db = pickle.load(f)
        new_db, merged = group_unknown(unknown_db, similarity_threshold=threshold)
        with open(db_path, 'wb') as f:
            pickle.dump(new_db, f)
        return jsonify({'status':'ok','merged':int(merged),'total':len(new_db)})
    except FileNotFoundError:
        return jsonify({'status':'ok','merged':0,'total':0})
    except Exception as e:
        return jsonify({'status':'error','error':str(e)}), 500

@app.route('/send_daily_now', methods=['POST'])
def send_daily_now():
    try:
        from shared import daily_known, daily_unknown, reset_daily_stats
        from utils.telegram_alert import send_daily_report_AI
        from datetime import datetime
        date_str = datetime.now().strftime('%Y-%m-%d')
        known = {k: list(v) for k, v in dict(daily_known).items()}
        unknown = {k: list(v) for k, v in dict(daily_unknown).items()}
        send_daily_report_AI(date_str, known, unknown)
        reset_daily_stats()
        return jsonify({'status':'ok'})
    except Exception as e:
        return jsonify({'status':'error','error':str(e)}), 500


@app.route('/restart_camera', methods=['POST'])
def restart_camera():
    """Перезапуск одной камеры по ключу из config CAMERAS (напр. cam2) или по индексу ?id=1."""
    try:
        global global_processes, global_stop_flags, frame_queue
        # Determine cam index
        config = load_config()
        cam_key = request.args.get('cam')
        cam_id_param = request.args.get('id')
        cam_index = None
        if cam_key and isinstance(config.get('CAMERAS'), dict):
            keys = list(config['CAMERAS'].keys())
            if cam_key in keys:
                cam_index = keys.index(cam_key)
            else:
                return jsonify({'status':'error','error':f'unknown cam key {cam_key}'}), 400
        elif cam_id_param is not None:
            try:
                cam_index = int(cam_id_param)
            except ValueError:
                return jsonify({'status':'error','error':'id must be int'}), 400
        else:
            return jsonify({'status':'error','error':'specify ?cam=camKey or ?id=index'}), 400

        if cam_index is None or cam_index < 0 or cam_index >= len(CAMERA_SOURCES):
            return jsonify({'status':'error','error':'invalid camera index'}), 400

        if not global_processes or not global_stop_flags:
            return jsonify({'status':'error','error':'cameras are not running'}), 400

        # Stop old camera process (avoid stopping face process which is last)
        try:
            old_flag = global_stop_flags[cam_index]
            old_proc = global_processes[cam_index]
            old_flag.value = True
            old_proc.join(timeout=5)
        except Exception as e:
            pass

        # Start new process for this camera
        from multiprocessing import Value, Process
        from camera_worker import camera_loop
        new_flag = Value('b', False)
        source = CAMERA_SOURCES[cam_index]
        p = Process(target=camera_loop, args=(source, frame_queue, cam_index, new_flag))
        p.start()

        # Replace in lists
        global_stop_flags[cam_index] = new_flag
        global_processes[cam_index] = p

        return jsonify({'status':'ok','restarted':cam_index})
    except Exception as e:
        return jsonify({'status':'error','error':str(e)}), 500