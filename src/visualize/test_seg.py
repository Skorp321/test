import cv2
import numpy as np
from ultralytics import SAM, YOLO
from collections import deque

# Инициализация SAM для сегментации
sam_model = SAM("sam2_b.pt")  # Используем версию 'sam_b' для баланса скорости и точности

# Инициализация YOLO для трекинга (опционально, если нужен гибридный подход)
tracker = YOLO('yolov8n.pt')  # Для трекинга можно использовать YOLO

# Загрузка первого кадра
cap = cv2.VideoCapture("data/video2.mp4")
ret, first_frame = cap.read()

# Пример точки (можно выбрать через интерфейс)
x, y = 300, 200  # Замените на свои координаты
# Правильный формат входных данных:
input_point = np.array([[x, y]], dtype=np.float32)  # Форма (1, 2)
input_label = np.array([1], dtype=np.int64)         # Форма (1,)

# Сегментация первого кадра
results = sam_model.predict(
    source=first_frame,
    points=input_point,  # Передаем массив, а не список
    labels=input_label,  # Передаем массив, а не список
)

# Извлечение маски
mask = results[0].masks[0].data.cpu().numpy().squeeze()

# Получение bounding box из маски
y_indices, x_indices = np.where(mask > 0)
bbox = (x_indices.min(), y_indices.min(), x_indices.max(), y_indices.max())

# Инициализация трекера (например, через YOLO)
tracker = YOLO('yolov8n.pt')
tracker.track(first_frame, persist=True, classes=[0], bboxes=[bbox])  # Пример для человека (class=0)

# Для сглаживания перемещений
track_history = deque(maxlen=30)

ret, prev_frame = cap.read()

prev_points = cv2.goodFeaturesToTrack(
    cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    # Трекинг объекта (через YOLO)
    tracked_results = tracker.track(frame, persist=True, verbose=False)
    
    if tracked_results[0].boxes:
        # Получение текущего bbox
        current_bbox = tracked_results[0].boxes.xyxy[0].cpu().numpy()
        track_history.append(current_bbox)

        # Уточнение маски через SAM
        refined_results = sam_model.predict(frame, bboxes=[current_bbox])
        refined_mask = refined_results[0].masks[0].data.cpu().numpy().squeeze()

        # Визуализация
        frame = cv2.rectangle(frame, 
                            (int(current_bbox[0]), int(current_bbox[1])),
                            (int(current_bbox[2]), int(current_bbox[3])),
                            (0, 255, 0), 2)
        frame[refined_mask > 0] = frame[refined_mask > 0] * 0.5 + [0, 0, 255] * 0.5

    cv2.imshow("Result", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()