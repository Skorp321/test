# YOLOv11n Detection на Raspberry Pi 4

Это приложение выполняет детекцию объектов с помощью YOLOv11n на Raspberry Pi 4, используя NCNN для оптимизации производительности.

## Требования

- Raspberry Pi 4
- Камера
- Docker
- Модель YOLOv11n в формате NCNN

## Установка

1. Убедитесь, что у вас установлен Docker:
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

2. Соберите Docker образ:
```bash
docker build -t yolo-detection .
```

## Запуск

1. Запустите контейнер с доступом к камере:
```bash
docker run --device=/dev/video0:/dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix yolo-detection
```

## Использование

- Приложение будет отображать видео с камеры с наложенными детекциями
- FPS ограничен до 8 кадров в секунду
- Для выхода нажмите 'q'

## Структура проекта

- `main.py` - основной файл приложения
- `Dockerfile` - файл для сборки Docker образа
- `requirements.txt` - зависимости Python
- `yolo11n_ncnn_model/` - директория с моделью NCNN 