# w1.py — теперь это модуль распознавания

import os
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

# Пути
MODEL_PATH = 'checkpoints/GN_W1.3_S1_ArcFace_epoch46.h5'
EMBEDDINGS_FILE = 'saved_embeddings.pkl'
DATABASE_DIR = 'faces_database'
THRESHOLD = 0.4
INPUT_SIZE = (112, 112)


class FaceRecognizer:
    def __init__(self, model_path=MODEL_PATH, db_dir=DATABASE_DIR, cache_file=EMBEDDINGS_FILE, threshold=THRESHOLD):
        self.threshold = threshold

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
        self.database = self.create_database(db_dir, cache_file)
        print(f"✅ База данных загружена: {len(self.database)} человек")

    def preprocess(self, img):
        """Подготавливает изображение под модель"""
        img = cv2.resize(img, INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def create_database(self, db_dir, cache_file):
        """Создаёт или загружает базу эмбеддингов"""
        if os.path.exists(cache_file):
            print(f"🔁 Загрузка закэшированной базы: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        print(f"📁 Создание базы из: {db_dir}")
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
                embedding = self.model.predict(processed, verbose=0)[0]
                embeddings.append(embedding)

            if embeddings:
                database[person_name] = embeddings
                print(f"👤 {person_name}: {len(embeddings)} фото")

        # Сохраняем кэш
        with open(cache_file, 'wb') as f:
            pickle.dump(database, f)
        print(f"💾 База сохранена в {cache_file}")
        return database

    def recognize(self, face_img):
        """
        Принимает изображение лица (BGR), возвращает (имя, схожесть)
        """
        try:
            processed = self.preprocess(face_img)
            emb = self.model.predict(processed, verbose=0)[0].reshape(1, -1)

            best_name = "Неизвестно"
            max_sim = 0.0

            for name, known_embs in self.database.items():
                sims = [cosine_similarity(emb, e.reshape(1, -1))[0][0] for e in known_embs]
                avg_sim = np.mean(sims)
                if avg_sim > max_sim:
                    max_sim = avg_sim
                    if avg_sim >= self.threshold:
                        best_name = name

            return best_name, max_sim
        except Exception as e:
            print(f"Ошибка при распознавании: {e}")
            return "Ошибка", 0.0