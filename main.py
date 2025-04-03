import cv2
import numpy as np
import time
from pathlib import Path
import ncnn
import torch

class YOLODetector:
    def __init__(self, model_path):
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = True
        self.net.load_param(str(Path(model_path) / "yolo11n_ncnn.param"))
        self.net.load_model(str(Path(model_path) / "yolo11n_ncnn.bin"))
        
        self.input_size = (640, 640)
        self.mean = [0, 0, 0]
        self.std = [255, 255, 255]
        
    def preprocess(self, img):
        img = cv2.resize(img, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - np.array(self.mean)) / np.array(self.std)
        img = img.transpose(2, 0, 1)
        return img

    def detect(self, img):
        img = self.preprocess(img)
        
        ex = self.net.create_extractor()
        ex.input("input", img)
        ret, out = ex.extract("output")
        
        # Обработка выходных данных
        detections = []
        if ret == 0:
            out = np.array(out)
            # Преобразование выходных данных в формат [x1, y1, x2, y2, conf, class]
            boxes = out.reshape(-1, 6)
            for box in boxes:
                if box[4] > 0.5:  # Порог уверенности
                    detections.append(box)
        
        return detections

def draw_detections(img, detections):
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def main():
    # Инициализация камеры
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Инициализация детектора
    detector = YOLODetector("yolo11n_ncnn_model")
    
    # Настройка FPS
    target_fps = 8
    frame_time = 1.0 / target_fps
    
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
            
        # Детекция объектов
        detections = detector.detect(frame)
        
        # Отрисовка результатов
        frame = draw_detections(frame, detections)
        
        # Отображение FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Показ результата
        cv2.imshow("YOLO Detection", frame)
        
        # Ожидание для поддержания целевого FPS
        elapsed_time = time.time() - start_time
        if elapsed_time < frame_time:
            time.sleep(frame_time - elapsed_time)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 