// src/yolo_ncnn_service.cpp
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "net.h"
#include "httplib.h"
#include <mutex>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

// Глобальные переменные
std::string model_path = "../models/";
std::string param_file = "yolo11n.param";
std::string bin_file = "yolo11n.bin";
std::vector<std::string> class_names;
std::mutex model_mutex;
ncnn::Net yolo;

// Загрузка имен классов
bool load_class_names(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening class names file: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty())
        {
            class_names.push_back(line);
        }
    }
    
    return true;
}

// Инициализация модели
bool init_model()
{
    try
    {
        // Загрузка имен классов
        if (!load_class_names(model_path + "coco.names"))
        {
            return false;
        }
        
        // Загрузка модели
        std::lock_guard<std::mutex> lock(model_mutex);
        
        // Включение Vulkan если возможно
        ncnn::create_gpu_instance();
        yolo.opt.use_vulkan_compute = true;
        
        int ret = yolo.load_param((model_path + param_file).c_str());
        if (ret != 0)
        {
            std::cerr << "Error loading NCNN param file" << std::endl;
            return false;
        }
        
        ret = yolo.load_model((model_path + bin_file).c_str());
        if (ret != 0)
        {
            std::cerr << "Error loading NCNN bin file" << std::endl;
            return false;
        }
        
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception during model initialization: " << e.what() << std::endl;
        return false;
    }
}

// Освобождение ресурсов
void cleanup()
{
    std::lock_guard<std::mutex> lock(model_mutex);
    yolo.clear();
    ncnn::destroy_gpu_instance();
}

// Детекция объектов
std::vector<Object> detect_objects(const cv::Mat& bgr)
{
    std::vector<Object> objects;
    
    // Преобразование изображения для использования в YOLO
    const int target_size = 640;
    int img_w = bgr.cols;
    int img_h = bgr.rows;
    
    // Вычисление соотношения сторон
    float scale = 1.0f;
    if (img_w > img_h)
    {
        scale = (float)target_size / img_w;
    }
    else
    {
        scale = (float)target_size / img_h;
    }
    
    const int target_w = img_w * scale;
    const int target_h = img_h * scale;
    
    // Преобразование размера с сохранением пропорций
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, img_w, img_h, target_w, target_h);
    
    // Нормализация
    const float mean_vals[3] = {0.0f, 0.0f, 0.0f};
    const float norm_vals[3] = {1/255.0f, 1/255.0f, 1/255.0f};
    in.substract_mean_normalize(mean_vals, norm_vals);
    
    // Создание padding для получения квадратного изображения
    ncnn::Mat in_pad;
    int wpad = (target_size - target_w) / 2;
    int hpad = (target_size - target_h) / 2;
    ncnn::copy_make_border(in, in_pad, hpad, target_size - hpad - target_h, wpad, target_size - wpad - target_w, ncnn::BORDER_CONSTANT, 0.0f);
    
    {
        std::lock_guard<std::mutex> lock(model_mutex);
        
        // Создание экстрактора NCNN
        ncnn::Extractor ex = yolo.create_extractor();
        
        // Устанавливаем input blob
        ex.input("images", in_pad);
        
        // Получаем выходной blob (имя может отличаться в зависимости от модели)
        ncnn::Mat out;
        ex.extract("output", out);
        
        // Разбор выходных данных (зависит от формата выхода модели)
        for (int i = 0; i < out.h; i++)
        {
            const float* values = out.row(i);
            
            Object obj;
            obj.label = values[0];
            obj.prob = values[1];
            
            // Координаты bbox (преобразование из нормализованных координат)
            float x1 = (values[2] - wpad) / scale;
            float y1 = (values[3] - hpad) / scale;
            float x2 = (values[4] - wpad) / scale;
            float y2 = (values[5] - hpad) / scale;
            
            // Ограничиваем границы, чтобы они не выходили за размеры изображения
            x1 = std::max(0.0f, std::min(img_w - 1.0f, x1));
            y1 = std::max(0.0f, std::min(img_h - 1.0f, y1));
            x2 = std::max(0.0f, std::min(img_w - 1.0f, x2));
            y2 = std::max(0.0f, std::min(img_h - 1.0f, y2));
            
            obj.rect.x = x1;
            obj.rect.y = y1;
            obj.rect.width = x2 - x1;
            obj.rect.height = y2 - y1;
            
            // Добавляем объект в результаты, если вероятность выше порога
            if (obj.prob > 0.25f)
            {
                objects.push_back(obj);
            }
        }
    }
    
    return objects;
}

// Обработка изображения и возврат результатов в JSON
json process_image(const cv::Mat& image)
{
    json result;
    result["success"] = true;
    result["detections"] = json::array();
    
    try
    {
        std::vector<Object> objects = detect_objects(image);
        
        for (const auto& obj : objects)
        {
            json detection;
            detection["label"] = obj.label < class_names.size() ? class_names[obj.label] : "unknown";
            detection["confidence"] = obj.prob;
            detection["x"] = obj.rect.x;
            detection["y"] = obj.rect.y;
            detection["width"] = obj.rect.width;
            detection["height"] = obj.rect.height;
            
            result["detections"].push_back(detection);
        }
    }
    catch (const std::exception& e)
    {
        result["success"] = false;
        result["error"] = e.what();
    }
    
    return result;
}

int main()
{
    // Инициализация модели
    if (!init_model())
    {
        std::cerr << "Failed to initialize YOLO model" << std::endl;
        return -1;
    }
    
    std::cout << "YOLO11n NCNN model loaded successfully" << std::endl;
    std::cout << "Starting HTTP server on port 5000" << std::endl;
    
    // Создание HTTP сервера
    httplib::Server svr;
    
    // Эндпоинт для проверки статуса
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    
    // Эндпоинт для обработки изображений
    svr.Post("/detect", [](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_file("image"))
        {
            res.status = 400;
            res.set_content("{\"error\":\"No image file provided\"}", "application/json");
            return;
        }
        
        const auto& file = req.get_file_value("image");
        
        // Декодирование изображения из бинарных данных
        std::vector<uchar> buffer(file.content.begin(), file.content.end());
        cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
        
        if (image.empty())
        {
            res.status = 400;
            res.set_content("{\"error\":\"Invalid image format\"}", "application/json");
            return;
        }
        
        // Обработка изображения
        json result = process_image(image);
        res.set_content(result.dump(), "application/json");
    });
    
    // Запуск сервера
    svr.listen("0.0.0.0", 5000);
    
    // Очистка ресурсов при завершении
    cleanup();
    
    return 0;
}