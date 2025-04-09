# Базовый образ для Raspberry Pi (ARM архитектура)
FROM arm64v8/python:3.9-slim

# Установка зависимостей системы
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    python3-opencv \
    libprotobuf-dev \
    protobuf-compiler \
    libvulkan-dev \
    vulkan-tools \
    libglfw3-dev \
    libgles2-mesa-dev \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Клонирование и сборка ncnn
RUN git clone https://github.com/Tencent/ncnn.git && \
    cd ncnn && \
    git submodule update --init && \
    mkdir -p build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DNCNN_VULKAN=ON \
          -DNCNN_SYSTEM_GLSLANG=OFF \
          -DNCNN_BUILD_EXAMPLES=ON \
          -DNCNN_PYTHON=ON \
          -DNCNN_BUILD_TOOLS=ON .. && \
    make -j$(nproc) && \
    make install && \
    cd /app/ncnn/python && \
    pip install -e .

# Установка дополнительных Python-зависимостей
RUN pip install --no-cache-dir numpy pillow

# Копирование кода детектора
COPY camera_detector.py /app/

# Создание volume для сохранения результатов
VOLUME /app/output

# Запуск детектора при старте контейнера
CMD ["python", "camera_detector.py"]