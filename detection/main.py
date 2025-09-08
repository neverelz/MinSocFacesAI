from yolo11 import AdvancedPersonDetector

def main():
    try:
        detector = AdvancedPersonDetector(confidence_threshold=0.7)
        detector.run()
    except Exception as error:
        print(f"Ошибка: {error}")


if __name__ == "__main__":
    main()


