#!/usr/bin/env python3
# src/client.py
import argparse
import requests
import cv2
import numpy as np
import json
import time
from PIL import Image, ImageDraw, ImageFont

def draw_detections(image_path, detections):
    # Загружаем изображение
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Попытка загрузить шрифт, если не получилось - используем стандартный
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    # Цвета для разных классов (цикличные)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)]
    
    # Рисуем каждую обнаруженную область
    for det in detections:
        x, y = det['x'], det['y']
        width, height = det['width'], det['height']
        label = det['label']
        confidence = det['confidence']
        
        # Вычисляем цвет для класса
        color_idx = hash(label) % len(colors)
        color = colors[color_idx]
        
        # Рисуем прямоугольник
        draw.rectangle([x, y, x + width, y + height], outline=color, width=3)
        
        # Рисуем текст
        text = f"{label}: {confidence:.2f}"
        draw.rectangle([x, y, x + len(text) * 8, y + 20], fill=color)
        draw.text((x + 2, y + 2), text, fill=(255, 255, 255), font=font)
    
    # Сохраняем результат
    output_path = image_path.rsplit('.', 1)[0] + "_detected.jpg"
    image.save(output_path)
    print(f"Результат сохранен в {output_path}")
    return output_path

def detect_objects(server_url, image_path):
    try:
        url = f"{server_url}/detect"
        
        # Подготовка файла изображения для отправки
        files = {'image': open(image_path, 'rb')}
        
        # Засекаем время
        start_time = time.time()
        
        # Отправляем запрос
        response = requests.post(url, files=files)
        
        # Закрываем файл
        files['image'].close()
        
        # Рассчитываем затраченное время
        elapsed_time = time.time() - start_time
        
        # Проверяем статус-код ответа
        if response.status_code != 200:
            print(f"Ошибка: {response.status_code} - {response.text}")
            return None
        
        # Парсим ответ
        result = response.json()
        
        if not result.get('success', False):
            print(f"Ошибка в обработке изображения: {result.get('error', 'Неизвестная ошибка')}")
            return None
        
        # Выводим информацию о детекции
        detections = result.get('detections', [])
        print(f"Обнаружено {len(detections)} объектов за {elapsed_time:.3f} секунд:")
        
        for i, det in enumerate(detections):
            print(f"{i+1}. {det['label']} (уверенность: {det['confidence']:.2f}) - "
                  f"позиция: ({det['x']:.1f}, {det['y']:.1f}), "
                  f"размеры: {det['width']:.1f}x{det['height']:.1f}")
        
        # Рисуем обнаружения на изображении
        if detections:
            output_path = draw_detections(image_path, detections)
            print(f"Визуализация сохранена в {output_path}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Ошибка соединения: {e}")
        return None
    except Exception as e:
        print(f"Ошибка: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Клиент для YOLO NCNN сервиса детекции объектов")
    parser.add_argument("--server", default="http://localhost:5000", help="URL сервера (по умолчанию: http://localhost:5000)")
    parser.add_argument("--image", required=True, help="Путь к изображению для анализа")
    
    args = parser.parse_args()
    
    print(f"Отправка изображения {args.image} на сервер {args.server}")
    detect_objects(args.server, args.image)

if __name__ == "__main__":
    main()