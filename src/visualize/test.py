from ultralytics import SAM
import cv2
from ultralytics.models.sam import SAM2VideoPredictor

# Create SAM2VideoPredictor
overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="sam2_b.pt")
predictor = SAM2VideoPredictor(overrides=overrides)

# Загрузка модели
#model = SAM("sam2.1_b.pt")

# Отображение информации о модели (опционально)
#model.info()

# Функция для обработки и отображения видео
def process_and_display_video(video_path):
    # Открытие видеофайла
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видеофайл {video_path}.")
        return

    # Получение параметров видео (FPS, размеры)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Создание окна для отображения видео
    cv2.namedWindow("Segmented Video", cv2.WINDOW_NORMAL)

    while True:
        # Чтение кадра из видео
        ret, frame = cap.read()
        if not ret:
            break  # Выход из цикла, если видео закончилось

        # Запуск инференса на текущем кадре
        results = predictor(frame)[0]

        # Получение размеченного изображения
        segmented_frame = results.plot()  # Предполагается, что модель возвращает объект с методом plot()

        # Отображение размеченного кадра
        cv2.imshow("Segmented Video", segmented_frame)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

# Путь к входному видео
video_path = "data/video1.mp4"

# Обработка и отображение видео
process_and_display_video(video_path)