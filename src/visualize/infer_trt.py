import cv2
import torch
from ultralytics import YOLO

def main():

    model = YOLO('yolov8n.pt')

    model.export(format='engine') 
    
    # Загрузка конвертированной модели TensorRT
    try:
        trt_model = YOLO('yolov8n.engine')
    except:
        print("Не удалось загрузить TensorRT модель. Используется оригинальная модель.")
        trt_model = model

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось получить кадр")
            break

        results = trt_model(frame)
        annotated_frame = results[0].plot()

        cv2.imshow('YOLOv8 TensorRT Inference', annotated_frame)
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()