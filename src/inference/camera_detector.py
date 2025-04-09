import cv2
import time
import signal
import sys
from ncnn.model_zoo import get_model
from ncnn.utils import draw_detection_objects

# Обработчик для корректного завершения при Ctrl+C
def signal_handler(sig, frame):
    print('Завершение работы...')
    if 'cap' in globals() and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    # Инициализация камеры Raspberry Pi
    # Вы можете изменить индекс (0), если у вас несколько камер
    print("Инициализация камеры...")
    cap = cv2.VideoCapture(0)
    
    # Проверка успешного открытия камеры
    if not cap.isOpened():
        print("Не удалось открыть камеру!")
        sys.exit(1)
        
    # Настройка разрешения камеры (по желанию)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Инициализация модели YOLOv8
    print("Загрузка модели YOLOv8s...")
    net = get_model(
        "yolov8s",
        target_size=640,
        prob_threshold=0.25,
        nms_threshold=0.45,
        num_threads=4,
        use_gpu=True,  # Установите False, если GPU недоступен на вашем Raspberry Pi
    )
    
    print("Запуск детектирования с камеры...")
    fps_time = time.time()
    frame_count = 0
    
    try:
        while True:
            # Захват кадра с камеры
            ret, frame = cap.read()
            if not ret:
                print("Не удалось получить кадр с камеры")
                break
            
            # Выполнение инференса
            objects = net(frame)
            
            # Отрисовка результатов детектирования
            result_frame = frame.copy()
            draw_detection_objects(result_frame, net.class_names, objects)
            
            # Расчет FPS
            frame_count += 1
            if frame_count >= 10:
                elapsed = time.time() - fps_time
                fps = frame_count / elapsed
                print(f"FPS: {fps:.2f}")
                fps_time = time.time()
                frame_count = 0
                
            # Отображение FPS на кадре
            cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Сохранение результата в файл (если графический интерфейс недоступен)
            cv2.imwrite("latest_detection.jpg", result_frame)
            
            # Отображение результата (если есть графический интерфейс)
            cv2.imshow("YOLOv8 Detection", result_frame)
            
            # Выход при нажатии клавиши 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        # Освобождение ресурсов
        cap.release()
        cv2.destroyAllWindows()
        print("Работа завершена")