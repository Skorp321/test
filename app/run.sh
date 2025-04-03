#!/bin/bash

# Скрипт для подготовки и запуска YOLO NCNN сервиса на Raspberry Pi

# Создаем необходимые директории
mkdir -p models
mkdir -p src/include
mkdir -p data

# Проверяем наличие модели
if [ ! -f "models/yolo11n.param" ] || [ ! -f "models/yolo11n.bin" ]; then
    echo "Модель YOLO11n не найдена в директории models/"
    echo "Пожалуйста, загрузите файлы модели yolo11n.param и yolo11n.bin в директорию models/"
    exit 1
fi

# Проверяем наличие файла с именами классов COCO
if [ ! -f "models/coco.names" ]; then
    echo "Загружаем файл с именами классов COCO..."
    wget -O models/coco.names https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
fi

# Сборка и запуск через Docker Compose
echo "Запускаем сборку и развертывание Docker контейнера..."
docker-compose up -d --build

# Проверка статуса сервиса
echo "Ожидаем запуск сервиса..."
sleep 10

# Проверяем доступность сервиса
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/health)

if [ "$response" = "200" ]; then
    echo "Сервис успешно запущен и доступен по адресу http://localhost:5000"
    echo "Используйте Python-клиент для тестирования:"
    echo "python3 src/client.py --image /path/to/your/image.jpg"
else
    echo "Ошибка: Сервис не отвечает. Проверьте логи Docker:"
    echo "docker-compose logs"
fi