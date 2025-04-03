# src/CMakeLists.txt
cmake_minimum_required(VERSION 3.0)
project(yolo_ncnn_service)

# C++11 поддержка
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Найти пакеты
find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

# Включаем каталоги для дополнительных библиотек
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Скачиваем зависимости
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/include/httplib.h)
    file(DOWNLOAD 
        https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/httplib.h
        SHOW_PROGRESS
    )
endif()

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/include/nlohmann/json.hpp)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/nlohmann)
    file(DOWNLOAD 
        https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/nlohmann/json.hpp
        SHOW_PROGRESS
    )
endif()

# Добавляем исполняемый файл
add_executable(yolo_ncnn_service yolo_ncnn_service.cpp)

# Линкуем библиотеки
target_link_libraries(yolo_ncnn_service 
    ${OpenCV_LIBS}
    ncnn
    pthread
)