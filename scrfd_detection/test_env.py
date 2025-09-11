try:
    import tensorflow as tf

    print("✅ TensorFlow импортирован")
    print("📌 Версия:", tf.__version__)
    print("📌 Расположение:", tf.__file__)

    from tensorflow.keras.models import load_model

    print("✅ Keras доступен")
except ModuleNotFoundError as e:
    print("❌ Модуль не найден:", e)
except AttributeError as e:
    print("❌ Атрибут не найден:", e)
    print("📌 Возможно, повреждённый импорт")
except Exception as e:
    print("❌ Другая ошибка:", e)