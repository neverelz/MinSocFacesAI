# recognizer.py

import os
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from collections import defaultdict

MODEL_PATH = 'checkpoints/GN_W1.3_S1_ArcFace_epoch46.h5'
EMBEDDINGS_FILE = 'saved_embeddings.pkl'
DATABASE_DIR = 'faces_database'
THRESHOLD = 0.5
INPUT_SIZE = (112, 112)

class FaceRecognizer:
    def __init__(self, model_path=MODEL_PATH, db_dir=DATABASE_DIR, cache_file=EMBEDDINGS_FILE, threshold=THRESHOLD, force_rebuild=False, use_gpu=True):
        self.threshold = threshold
        self.db_dir = db_dir
        self.cache_file = cache_file
        self.use_gpu = use_gpu

        # Настройка TensorFlow под CPU/GPU
        if not use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Отключаем GPU для TF
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
            # Оптимизация для CPU
            tf.config.threading.set_intra_op_parallelism_threads(0)  # auto
            tf.config.threading.set_inter_op_parallelism_threads(0)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        self.model = load_model(model_path)
        print("✅ Модель распознавания загружена")

        # Тест
        test_input = np.random.rand(1, 112, 112, 3).astype(np.float32) / 255.0
        test_emb = self.model.predict(test_input, verbose=0)
        if np.any(np.isnan(test_emb)) or np.allclose(test_emb, 0):
            raise ValueError("Модель выдает некорректные эмбеддинги")
        print(f"✅ Модель работает: эмбеддинг размером {test_emb.shape[1]}")

        self.database = self.create_database(db_dir, cache_file, force_rebuild=force_rebuild)
        print(f"✅ База данных загружена: {len(self.database)} человек")
        self._rebuild_index()

    def preprocess(self, img):
        img = cv2.resize(img, INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img

    @staticmethod
    def l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm != 0 else vec

    def create_database(self, db_dir, cache_file, force_rebuild=False):
        if not force_rebuild and os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                try:
                    return pickle.load(f)
                except:
                    pass
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        database = {}
        for person_name in os.listdir(db_dir):
            person_path = os.path.join(db_dir, person_name)
            if not os.path.isdir(person_path):
                continue
            embeddings = []
            for filename in os.listdir(person_path):
                filepath = os.path.join(person_path, filename)
                img = cv2.imread(filepath)
                if img is None:
                    continue
                processed = self.preprocess(img)
                embedding = self.model.predict(np.expand_dims(processed, axis=0), verbose=0)[0]
                embedding = self.l2_normalize(embedding)
                embeddings.append(embedding)
            if embeddings:
                database[person_name] = embeddings
        with open(cache_file, 'wb') as f:
            pickle.dump(database, f)
        return database

    def _rebuild_index(self):
        all_embs, all_labels = [], []
        self._label_to_indices = defaultdict(list)
        idx = 0
        for label, embs in self.database.items():
            for e in embs:
                all_embs.append(e)
                all_labels.append(label)
                self._label_to_indices[label].append(idx)
                idx += 1
        if all_embs:
            self._all_embeddings = np.vstack(all_embs).astype(np.float32)
            norms = np.linalg.norm(self._all_embeddings, axis=1, keepdims=True) + 1e-8
            self._all_embeddings /= norms
        else:
            self._all_embeddings = np.empty((0, 0), dtype=np.float32)
        self._all_labels = all_labels

    def recognize_batch(self, face_imgs):
        if not face_imgs:
            return [], []

        # Подготовка батча
        processed = np.stack([self.preprocess(img) for img in face_imgs])
        embs = self.model.predict(processed, verbose=0).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        embs /= norms

        if self._all_embeddings.size == 0:
            return ["Неизвестно"] * len(face_imgs), [0.0] * len(face_imgs)

        sim_matrix = embs @ self._all_embeddings.T
        names, sims = [], []
        for i in range(len(face_imgs)):
            best_name, max_sim = "Неизвестно", 0.0
            for label, indices in self._label_to_indices.items():
                person_sim = float(np.max(sim_matrix[i, indices]))
                if person_sim > max_sim:
                    max_sim = person_sim
                    if person_sim >= self.threshold:
                        best_name = label
            names.append(best_name)
            sims.append(max_sim)
        return names, sims

    def get_next_person_id(self) -> str:
        if not os.path.exists(DATABASE_DIR):
            os.makedirs(DATABASE_DIR, exist_ok=True)
        max_id = 0
        for name in os.listdir(DATABASE_DIR):
            if os.path.isdir(os.path.join(DATABASE_DIR, name)) and name.isdigit():
                try:
                    max_id = max(max_id, int(name))
                except:
                    continue
        return str(max_id + 1)

    def _get_next_image_index(self, person_id: str) -> int:
        person_dir = os.path.join(DATABASE_DIR, person_id)
        if not os.path.exists(person_dir):
            return 1
        existing = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.png'))]
        nums = []
        for f in existing:
            base = os.path.splitext(f)[0]
            try:
                nums.append(int(base))
            except:
                continue
        return max(nums) + 1 if nums else 1

    def add_image_to_person(self, person_id: str, face_img_bgr: np.ndarray) -> str:
        person_dir = os.path.join(DATABASE_DIR, person_id)
        os.makedirs(person_dir, exist_ok=True)
        idx = self._get_next_image_index(person_id)
        filepath = os.path.join(person_dir, f"{idx:03d}.jpg")
        cv2.imwrite(filepath, face_img_bgr)

        processed = self.preprocess(face_img_bgr)
        embedding = self.model.predict(np.expand_dims(processed, axis=0), verbose=0)[0]
        embedding = self.l2_normalize(embedding)

        if person_id not in self.database:
            self.database[person_id] = []
        self.database[person_id].append(embedding)

        if self._all_embeddings.size == 0:
            self._all_embeddings = embedding.reshape(1, -1).astype(np.float32)
            self._all_labels = [person_id]
            self._label_to_indices = defaultdict(list)
            self._label_to_indices[person_id] = [0]
        else:
            new_idx = len(self._all_labels)
            self._all_embeddings = np.vstack([self._all_embeddings, embedding.astype(np.float32)])
            self._all_labels.append(person_id)
            self._label_to_indices[person_id].append(new_idx)

        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(self.database, f)
        return filepath

    def clear_database_and_cache(self):
        import shutil
        if os.path.exists(self.db_dir):
            shutil.rmtree(self.db_dir)
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        self.database = {}
        self._all_embeddings = None
        self._all_labels = []
        self._label_to_indices = defaultdict(list)
        os.makedirs(self.db_dir, exist_ok=True)
        print("🧹 База и кэш очищены.")