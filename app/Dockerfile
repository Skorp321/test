FROM ubuntu:20.04

# Предотвращение интерактивных запросов во время установки
ENV DEBIAN_FRONTEND=noninteractive

# Установка необходимых пакетов
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libprotobuf-dev \
    protobuf-compiler \
    libopencv-dev \
    python3 \
    python3-pip \
    wget \
    unzip \
    libvulkan-dev \
    python3-opencv

# Клонирование и установка NCNN
WORKDIR /opt
RUN git clone https://github.com/Tencent/ncnn.git
WORKDIR /opt/ncnn
RUN git submodule update --init
RUN mkdir -p build
WORKDIR /opt/ncnn/build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DNCNN_BUILD_EXAMPLES=ON ..
RUN make -j$(nproc)
RUN make install

# Создание рабочей директории для приложения
WORKDIR /app

# Копирование модели (предполагается, что модель уже есть в директории сборки)
COPY ./models /app/models

# Копирование исходного кода сервиса
COPY ./src /app/src

# Установка Python зависимостей
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Сборка сервиса
RUN mkdir -p build
WORKDIR /app/build
RUN cmake ../src
RUN make

# Открытие порта для API
EXPOSE 5000

# Запуск сервиса
CMD ["./yolo_ncnn_service"]