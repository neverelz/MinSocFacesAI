# w1.py — теперь это модуль распознавания

import os
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List
from collections import defaultdict

# Пути
MODEL_PATH = 'checkpoints/GN_W1.3_S1_ArcFace_epoch46.h5'
EMBEDDINGS_FILE = 'saved_embeddings.pkl'
DATABASE_DIR = 'faces_database'
THRESHOLD = 0.5
INPUT_SIZE = (112, 112)


class FaceRecognizer:
    def __init__(self, model_path=MODEL_PATH, db_dir=DATABASE_DIR, cache_file=EMBEDDINGS_FILE, threshold=THRESHOLD, force_rebuild: bool = False):
        self.threshold = threshold
        self.db_dir = db_dir
        self.cache_file = cache_file
        # Индекс для быстрого поиска
        self._all_embeddings = None  # np.ndarray [N, D]
        self._all_labels = []        # List[str] длиной N
        self._label_to_indices = defaultdict(list)

        # Загрузка модели
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        self.model = load_model(model_path)
        print("✅ Модель распознавания загружена")

        # Тест модели
        test_input = np.random.rand(1, 112, 112, 3).astype(np.float32) / 255.0
        test_emb = self.model.predict(test_input, verbose=0)
        if np.any(np.isnan(test_emb)) or np.allclose(test_emb, 0):
            raise ValueError("Модель выдает некорректные эмбеддинги")
        print(f"✅ Модель работает: эмбеддинг размером {test_emb.shape[1]}")

        # Создаем базу эмбеддингов
        self.database = self.create_database(self.db_dir, self.cache_file, force_rebuild=force_rebuild)
        print(f"✅ База данных загружена: {len(self.database)} человек")
        # Построить индекс
        self._rebuild_index()

    


    def preprocess(self, img):
        """Подготавливает изображение под модель"""
        img = cv2.resize(img, INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    @staticmethod
    def l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def create_database(self, db_dir, cache_file, force_rebuild: bool = False):
        """Создаёт или загружает базу эмбеддингов"""
        if not force_rebuild and os.path.exists(cache_file):
            print(f"🔁 Загрузка закэшированной базы: {cache_file}")
            with open(cache_file, 'rb') as f:
                try:
                    return pickle.load(f)
                except Exception:
                    print("⚠️ Не удалось прочитать кэш, пересоздаю базу...")
        # Если папки нет — создадим пустую
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        print(f"📁 Создание базы из: {db_dir}")
        database: Dict[str, List[np.ndarray]] = {}
        for person_name in os.listdir(db_dir):
            person_path = os.path.join(db_dir, person_name)
            if not os.path.isdir(person_path):
                continue
            embeddings: List[np.ndarray] = []
            for filename in os.listdir(person_path):
                filepath = os.path.join(person_path, filename)
                img = cv2.imread(filepath)
                if img is None:
                    continue
                processed = self.preprocess(img)
                embedding = self.model.predict(processed, verbose=0)[0]
                embedding = self.l2_normalize(embedding)
                embeddings.append(embedding)
            if embeddings:
                database[person_name] = embeddings
                print(f"👤 {person_name}: {len(embeddings)} фото")
        # Сохраняем кэш
        with open(cache_file, 'wb') as f:
            pickle.dump(database, f)
        print(f"💾 База сохранена в {cache_file}")
        return database

    def _rebuild_index(self):
        """Строит векторизованный индекс из текущей базы."""
        all_embs = []
        all_labels = []
        label_to_indices = defaultdict(list)
        idx = 0
        for label, embs in self.database.items():
            for e in embs:
                all_embs.append(e)
                all_labels.append(label)
                label_to_indices[label].append(idx)
                idx += 1
        if all_embs:
            self._all_embeddings = np.vstack(all_embs).astype(np.float32)
            # На всякий случай нормализуем
            norms = np.linalg.norm(self._all_embeddings, axis=1, keepdims=True) + 1e-8
            self._all_embeddings = self._all_embeddings / norms
        else:
            self._all_embeddings = np.empty((0, 0), dtype=np.float32)
        self._all_labels = all_labels
        self._label_to_indices = label_to_indices

    def recognize(self, face_img):
        """
        Принимает изображение лица (BGR), возвращает (имя, схожесть)
        """
        try:
            processed = self.preprocess(face_img)
            emb = self.model.predict(processed, verbose=0)[0]
            emb = self.l2_normalize(emb).reshape(1, -1)

            best_name = "Неизвестно"
            max_sim = 0.0

            if self._all_embeddings is not None and self._all_embeddings.size > 0:
                sims_all = emb @ self._all_embeddings.T  # [1, N]
                sims_all = sims_all.reshape(-1)
                # Агрегируем по label: берём максимум по каждому человеку
                for label, indices in self._label_to_indices.items():
                    if not indices:
                        continue
                    person_sim = float(np.max(sims_all[indices]))
                    if person_sim > max_sim:
                        max_sim = person_sim
                        if person_sim >= self.threshold:
                            best_name = label
            else:
                # База пуста
                best_name = "Неизвестно"
                max_sim = 0.0

            return best_name, max_sim
        except Exception as e:
            print(f"Ошибка при распознавании: {e}")
            return "Ошибка", 0.0

    def recognize_batch(self, face_imgs: List[np.ndarray]):
        """Пакетное распознавание. Возвращает списки имён и схожестей той же длины."""
        if not face_imgs:
            return [], []
        # Препроцесс батчем
        processed = []
        for img in face_imgs:
            img_r = cv2.resize(img, INPUT_SIZE)
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
            img_r = img_r.astype(np.float32) / 255.0
            processed.append(img_r)
        batch = np.stack(processed, axis=0)  # [B, 112, 112, 3]
        # Предсказание и нормализация
        embs = self.model.predict(batch, verbose=0)  # [B, D]
        embs = embs.astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        embs = embs / norms

        names = []
        sims = []
        if self._all_embeddings is None or self._all_embeddings.size == 0:
            # Пустая база
            names = ["Неизвестно"] * len(face_imgs)
            sims = [0.0] * len(face_imgs)
            return names, sims

        # Косинусные сходства батчем: [B, N]
        sim_matrix = embs @ self._all_embeddings.T
        for i in range(sim_matrix.shape[0]):
            sims_all = sim_matrix[i]
            best_name = "Неизвестно"
            max_sim = 0.0
            for label, indices in self._label_to_indices.items():
                if not indices:
                    continue
                person_sim = float(np.max(sims_all[indices]))
                if person_sim > max_sim:
                    max_sim = person_sim
                    if person_sim >= self.threshold:
                        best_name = label
            names.append(best_name)
            sims.append(max_sim)
        return names, sims

    def get_next_person_id(self) -> str:
        """Определяет следующий числовой ID в папке faces_database"""
        if not os.path.exists(DATABASE_DIR):
            os.makedirs(DATABASE_DIR, exist_ok=True)
        max_id = 0
        for name in os.listdir(DATABASE_DIR):
            path = os.path.join(DATABASE_DIR, name)
            if os.path.isdir(path) and name.isdigit():
                try:
                    max_id = max(max_id, int(name))
                except ValueError:
                    continue
        return str(max_id + 1)

    def _get_next_image_index(self, person_id: str) -> int:
        person_dir = os.path.join(DATABASE_DIR, person_id)
        if not os.path.exists(person_dir):
            return 1
        existing = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not existing:
            return 1
        nums = []
        for f in existing:
            base = os.path.splitext(f)[0]
            try:
                nums.append(int(base))
            except ValueError:
                try:
                    nums.append(int(base.strip().lstrip('0')))
                except Exception:
                    continue
        return (max(nums) + 1) if nums else 1

    def add_image_to_person(self, person_id: str, face_img_bgr: np.ndarray) -> str:
        """Сохраняет изображение лица в `faces_database/person_id` и обновляет базу эмбеддингов"""
        person_dir = os.path.join(DATABASE_DIR, person_id)
        os.makedirs(person_dir, exist_ok=True)

        idx = self._get_next_image_index(person_id)
        filename = f"{idx:03d}.jpg"
        filepath = os.path.join(person_dir, filename)

        # Сохранение изображения (в RGB или BGR — не критично, preprocess учитывает)
        cv2.imwrite(filepath, face_img_bgr)

        # Обновление эмбеддингов в памяти и кэша
        processed = self.preprocess(face_img_bgr)
        embedding = self.model.predict(processed, verbose=0)[0]
        embedding = self.l2_normalize(embedding)

        if person_id not in self.database:
            self.database[person_id] = []
        # Добавляем в базу и индекс
        self.database[person_id].append(embedding)
        # Обновляем индекс инкрементально
        if self._all_embeddings is None or self._all_embeddings.size == 0:
            self._all_embeddings = embedding.reshape(1, -1).astype(np.float32)
            self._all_labels = [person_id]
            self._label_to_indices = defaultdict(list)
            self._label_to_indices[person_id].append(0)
        else:
            self._all_embeddings = np.vstack([self._all_embeddings, embedding.astype(np.float32)])
            new_idx = len(self._all_labels)
            self._all_labels.append(person_id)
            self._label_to_indices[person_id].append(new_idx)

        # Обновляем кэш
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(self.database, f)

        return filepath

    def clear_database_and_cache(self):
        """Полностью очищает папку с базой лиц и кэш эмбеддингов."""
        import shutil
        # Удаляем папку с лицами
        if os.path.exists(self.db_dir):
            for name in os.listdir(self.db_dir):
                path = os.path.join(self.db_dir, name)
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                except Exception as e:
                    print(f"⚠️ Не удалось удалить {path}: {e}")
        else:
            os.makedirs(self.db_dir, exist_ok=True)
        # Удаляем кэш
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
            except Exception as e:
                print(f"⚠️ Не удалось удалить {self.cache_file}: {e}")
        # Очистить базу в памяти
        self.database = {}
        self._all_embeddings = None
        self._all_labels = []
        self._label_to_indices = defaultdict(list)
        print("🧹 База и кэш очищены.")