import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import json
import os
from ultralytics import SAM

class TkinterVideoAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Аннотатор видео")
        self.root.geometry("800x600")
        
        # Загрузка модели SAM
        try:
            self.sam_model = SAM("sam2_b.pt")
            self.status_var = tk.StringVar()
            self.status_var.set("Модель SAM успешно загружена")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель SAM: {e}")
            self.sam_model = None
            self.status_var = tk.StringVar()
            self.status_var.set("Ошибка загрузки модели SAM")
        
        # Переменные для хранения данных
        self.video_path = None
        self.frame = None
        self.points = {}  # {id: (x, y)}
        self.current_label = 1
        self.output_filename = None
        self.photo = None  # Для хранения PhotoImage
        self.segmented_frame = None  # Для хранения сегментированного кадра
        self.show_segmentation = False  # Флаг отображения сегментации
        
        # Переменные для навигации по видео
        self.cap = None  # Объект захвата видео
        self.current_frame_idx = 0  # Индекс текущего кадра
        self.total_frames = 0  # Общее количество кадров в видео
        
        # Создаем меню
        self.create_menu()
        
        # Создаем интерфейс
        self.create_widgets()
        
        # Привязываем события
        self.canvas.bind("<Button-1>", self.add_point)
        
        # Статусная строка
        self.statusbar = tk.Label(root, textvariable=self.status_var, 
                                 bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_menu(self):
        """Создание меню приложения"""
        menubar = tk.Menu(self.root)
        
        # Меню Файл
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Открыть видеофайл", command=self.load_video)
        filemenu.add_command(label="Сохранить точки", command=self.save_points)
        filemenu.add_command(label="Сохранить маску и bbox", command=self.save_segmentation)
        filemenu.add_separator()
        filemenu.add_command(label="Выход", command=self.root.quit)
        menubar.add_cascade(label="Файл", menu=filemenu)
        
        # Меню Навигация
        navmenu = tk.Menu(menubar, tearoff=0)
        navmenu.add_command(label="Следующий кадр", command=self.next_frame)
        navmenu.add_command(label="Предыдущий кадр", command=self.prev_frame)
        navmenu.add_command(label="Перейти к кадру...", command=self.goto_frame)
        menubar.add_cascade(label="Навигация", menu=navmenu)
        
        # Меню Правка
        editmenu = tk.Menu(menubar, tearoff=0)
        editmenu.add_command(label="Очистить все точки", command=self.clear_points)
        menubar.add_cascade(label="Правка", menu=editmenu)
        
        # Меню Сегментация
        segmenu = tk.Menu(menubar, tearoff=0)
        segmenu.add_command(label="Запустить сегментацию", command=self.run_segmentation)
        segmenu.add_command(label="Переключить режим отображения", command=self.toggle_segmentation)
        segmenu.add_command(label="Сохранить результаты сегментации", command=self.save_segmentation)
        menubar.add_cascade(label="Сегментация", menu=segmenu)
        
        # Меню Справка
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="О программе", command=self.show_about)
        menubar.add_cascade(label="Справка", menu=helpmenu)
        
        self.root.config(menu=menubar)
    
    def create_widgets(self):
        """Создание элементов интерфейса"""
        # Фрейм для кнопок
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Кнопки файловых операций
        open_button = tk.Button(button_frame, text="Открыть видео", command=self.load_video)
        open_button.pack(side=tk.LEFT, padx=5)
        
        save_button = tk.Button(button_frame, text="Сохранить точки", command=self.save_points)
        save_button.pack(side=tk.LEFT, padx=5)
        
        save_mask_button = tk.Button(button_frame, text="Сохранить маску", command=self.save_segmentation)
        save_mask_button.pack(side=tk.LEFT, padx=5)
        
        clear_button = tk.Button(button_frame, text="Очистить точки", command=self.clear_points)
        clear_button.pack(side=tk.LEFT, padx=5)
        
        # Кнопка сегментации
        segment_button = tk.Button(button_frame, text="Сегментировать", command=self.run_segmentation)
        segment_button.pack(side=tk.LEFT, padx=5)
        
        toggle_button = tk.Button(button_frame, text="Показать/скрыть сегментацию", command=self.toggle_segmentation)
        toggle_button.pack(side=tk.LEFT, padx=5)
        
        # Кнопки навигации
        nav_frame = tk.Frame(button_frame)
        nav_frame.pack(side=tk.RIGHT, padx=5)
        
        prev_frame_button = tk.Button(nav_frame, text="◄", command=self.prev_frame, width=3)
        prev_frame_button.pack(side=tk.LEFT)
        
        self.frame_label = tk.Label(nav_frame, text="Кадр: 0 / 0")
        self.frame_label.pack(side=tk.LEFT, padx=5)
        
        next_frame_button = tk.Button(nav_frame, text="►", command=self.next_frame, width=3)
        next_frame_button.pack(side=tk.LEFT)
        
        # Канвас для отображения изображения
        self.canvas_frame = tk.Frame(self.root, bd=2, relief=tk.SUNKEN)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Информационная панель
        info_frame = tk.Frame(self.root)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        tk.Label(info_frame, text="Инструкция: ").pack(side=tk.LEFT)
        tk.Label(info_frame, 
                text="Нажмите на изображение, чтобы добавить точку. Используйте кнопки навигации для перехода между кадрами.").pack(side=tk.LEFT)
    
    def load_video(self):
        """Загрузка видео и отображение первого кадра"""
        file_path = filedialog.askopenfilename(
            title="Выберите видеофайл",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        
        if not file_path:
            return
            
        try:
            # Закрываем предыдущий видеофайл, если был открыт
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            
            # Открываем новый видеофайл
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                messagebox.showerror("Ошибка", f"Не удалось открыть видеофайл: {file_path}")
                return
            
            # Получаем информацию о видео
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame_idx = 0
            
            # Читаем первый кадр
            ret, frame = self.cap.read()
            if not ret or frame is None:
                messagebox.showerror("Ошибка", "Не удалось прочитать кадр из видео")
                return
            
            # Сохраняем данные
            self.video_path = file_path
            self.frame = frame
            self.output_filename = os.path.splitext(file_path)[0] + "_points.json"
            
            # Очищаем точки при загрузке нового видео
            self.clear_points(confirm=False)
            
            # Обновляем метку с номером кадра
            self.update_frame_label()
            
            # Отображаем первый кадр
            self.display_frame()
            
            # Обновляем статус
            self.status_var.set(f"Загружен видеофайл: {os.path.basename(file_path)} ({self.total_frames} кадров)")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при загрузке видео: {e}")
    
    def next_frame(self):
        """Переход к следующему кадру"""
        if self.cap is None or not self.cap.isOpened():
            messagebox.showinfo("Информация", "Сначала загрузите видеофайл")
            return
        
        # Проверяем, достигли ли мы конца видео
        if self.current_frame_idx >= self.total_frames - 1:
            messagebox.showinfo("Информация", "Достигнут конец видео")
            return
        self.show_segmentation = False
        # Считываем следующий кадр
        ret, frame = self.cap.read()
        if not ret or frame is None:
            messagebox.showerror("Ошибка", "Не удалось прочитать следующий кадр")
            return
        
        # Обновляем данные
        self.current_frame_idx += 1
        self.frame = frame
        
        # Очищаем точки при переходе к новому кадру
        self.points = {}
        self.current_label = 1
        
        # Обновляем метку с номером кадра
        self.update_frame_label()
        
        # Отображаем новый кадр
        self.display_frame()
        
        # Обновляем статус
        self.status_var.set(f"Кадр {self.current_frame_idx + 1} из {self.total_frames}")
    
    def prev_frame(self):
        """Переход к предыдущему кадру"""
        if self.cap is None or not self.cap.isOpened():
            messagebox.showinfo("Информация", "Сначала загрузите видеофайл")
            return
        
        # Проверяем, находимся ли мы в начале видео
        if self.current_frame_idx <= 0:
            messagebox.showinfo("Информация", "Это первый кадр видео")
            return
        self.show_segmentation = False
        # Перемотка видео назад
        self.current_frame_idx -= 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        
        # Считываем кадр
        ret, frame = self.cap.read()
        if not ret or frame is None:
            messagebox.showerror("Ошибка", "Не удалось прочитать предыдущий кадр")
            return
        
        # Обновляем данные
        self.frame = frame
        
        # Очищаем точки при переходе к новому кадру
        self.points = {}
        self.current_label = 1
        
        # Обновляем метку с номером кадра
        self.update_frame_label()
        
        # Отображаем новый кадр
        self.display_frame()
        
        # Обновляем статус
        self.status_var.set(f"Кадр {self.current_frame_idx + 1} из {self.total_frames}")
    
    def goto_frame(self):
        """Переход к указанному кадру"""
        if self.cap is None or not self.cap.isOpened():
            messagebox.showinfo("Информация", "Сначала загрузите видеофайл")
            return
        
        # Запрашиваем номер кадра у пользователя
        frame_number = simpledialog.askinteger("Переход", 
                                             f"Введите номер кадра (1-{self.total_frames}):",
                                             minvalue=1, maxvalue=self.total_frames)
        
        if frame_number is None:
            return
        
        # Переходим к указанному кадру
        frame_idx = frame_number - 1  # Преобразуем в индекс (начиная с 0)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Считываем кадр
        ret, frame = self.cap.read()
        if not ret or frame is None:
            messagebox.showerror("Ошибка", f"Не удалось прочитать кадр {frame_number}")
            return
        
        # Обновляем данные
        self.current_frame_idx = frame_idx
        self.frame = frame
        
        # Очищаем точки при переходе к новому кадру
        self.points = {}
        self.current_label = 1
        
        # Обновляем метку с номером кадра
        self.update_frame_label()
        
        # Отображаем новый кадр
        self.display_frame()
        
        # Обновляем статус
        self.status_var.set(f"Кадр {self.current_frame_idx + 1} из {self.total_frames}")
    
    def update_frame_label(self):
        """Обновление метки с номером текущего кадра"""
        if hasattr(self, 'frame_label'):
            self.frame_label.config(text=f"Кадр: {self.current_frame_idx + 1} / {self.total_frames}")
    
    def display_frame(self):
        """Отображение кадра с точками на канвасе"""
        if self.frame is None:
            return
            
        # Выбираем кадр для отображения
        if self.show_segmentation and self.segmented_frame is not None:
            display_frame = self.segmented_frame.copy()
        else:
            # Создаем копию кадра для рисования
            display_frame = self.frame.copy()
            
            # Рисуем точки
            for label, (x, y) in self.points.items():
                cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(display_frame, str(label), (x + 10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Конвертируем изображение из BGR (OpenCV) в RGB (PIL)
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Получаем размеры канваса
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Масштабируем изображение, если нужно
        if canvas_width > 1 and canvas_height > 1:  # Убедимся, что канвас уже имеет размеры
            img_width, img_height = pil_image.size
            
            # Вычисляем масштаб
            scale_width = canvas_width / img_width
            scale_height = canvas_height / img_height
            scale = min(scale_width, scale_height)
            
            # Новые размеры
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Масштабируем изображение
            if scale < 1:  # Уменьшаем только если нужно
                pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Конвертируем в PhotoImage для отображения на канвасе
        self.photo = ImageTk.PhotoImage(image=pil_image)
        
        # Обновляем размер канваса и отображаем изображение
        self.canvas.config(width=pil_image.width, height=pil_image.height)
        self.canvas.delete("all")  # Очищаем канвас
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
    
    def add_point(self, event):
        """Добавление точки по клику мыши"""
        if self.frame is None:
            messagebox.showinfo("Информация", "Сначала загрузите видеофайл")
            return
        
        # Получаем координаты клика
        x, y = event.x, event.y
        
        # Проверяем масштабирование
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_height, img_width = self.frame.shape[:2]
        
        # Вычисляем масштаб
        scale_width = canvas_width / img_width
        scale_height = canvas_height / img_height
        scale = min(scale_width, scale_height)
        
        # Пересчитываем координаты с учетом масштаба
        if scale < 1:  # Если изображение уменьшено
            x = int(x / scale)
            y = int(y / scale)
        
        # Проверяем, что клик внутри изображения
        if x < 0 or y < 0 or x >= img_width or y >= img_height:
            return
            
        # Спрашиваем метку
        label = simpledialog.askinteger("Метка точки", 
                                       f"Введите числовую метку (текущая: {self.current_label}):",
                                       initialvalue=self.current_label,
                                       parent=self.root)
        
        if label is not None:
            self.current_label = label
            self.points[label] = (x, y)
            
            # Сбрасываем сегментацию при добавлении новой точки
            self.segmented_frame = None
            self.show_segmentation = False
            
            self.display_frame()
            self.status_var.set(f"Кадр {self.current_frame_idx + 1}: добавлена точка {label} в позиции ({x}, {y})")
    
    def save_points(self):
        """Сохранение точек в файл"""
        if not self.points:
            messagebox.showinfo("Информация", "Нет точек для сохранения")
            return
            
        if not self.output_filename:
            self.output_filename = filedialog.asksaveasfilename(
                title="Сохранить точки",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")]
            )
            
            if not self.output_filename:
                return
        
        try:
            # Создаем данные для сохранения
            output_data = {
                "frame_index": self.current_frame_idx,
                "frame_number": self.current_frame_idx + 1,
                "video_path": self.video_path,
                "points": {str(label): {"x": x, "y": y} for label, (x, y) in self.points.items()}
            }
            
            with open(self.output_filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4)
                
            messagebox.showinfo("Успешно", f"Точки сохранены в файл:\n{self.output_filename}")
            self.status_var.set(f"Точки для кадра {self.current_frame_idx + 1} сохранены в файл: {os.path.basename(self.output_filename)}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{e}")
    
    def clear_points(self, confirm=True):
        """Очистка всех точек"""
        if not self.points:
            return
            
        if confirm:
            if not messagebox.askyesno("Подтверждение", "Вы уверены, что хотите удалить все точки?"):
                return
                
        self.points = {}
        self.current_label = 1
        self.segmented_frame = None  # Сбрасываем сегментацию
        self.show_segmentation = False
        self.display_frame()
        self.status_var.set("Все точки удалены")
    
    def show_about(self):
        """Отображение информации о программе"""
        messagebox.showinfo("О программе", 
                          "Аннотатор видео\n\n"
                          "Программа для отметки точек на кадрах видео\n"
                          "и сохранения их координат в JSON файл.")
    
    def on_resize(self, event):
        """Обработка изменения размера окна"""
        # Перерисовываем кадр при изменении размера окна
        self.display_frame()
    
    def on_closing(self):
        """Обработка закрытия окна"""
        # Закрываем видеофайл, если открыт
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()
    
    def run_segmentation(self):
        """Запуск сегментации SAM для текущего кадра с выбранными точками"""
        if self.frame is None:
            messagebox.showinfo("Информация", "Сначала загрузите видеофайл")
            return
            
        if not self.points:
            messagebox.showinfo("Информация", "Сначала добавьте хотя бы одну точку")
            return
            
        if self.sam_model is None:
            messagebox.showerror("Ошибка", "Модель SAM не загружена")
            return
            
        try:
            # Подготовка точек для SAM
            prompt_points = []
            labels = []
            for label, (x, y) in self.points.items():
                prompt_points.append([x, y])
                labels.append(label)

            
            # Запуск инференса SAM с выбранными точками
            results = self.sam_model.predict(
                source=self.frame.copy(),  # Используем копию кадра
                points=prompt_points,  # Формат массива: [B, N, 2], где B=1, N - количество точек
                labels=labels,  # Все точки как foreground (1)
                show=False,
                save=False
            )
            
            # Получаем первый результат
            if len(results) > 0:
                self.segmentation_result = results[0]  # Сохраняем результат для последующего сохранения
                # Получаем визуализацию результата сегментации
                self.segmented_frame = self.segmentation_result.plot()
                # Отображаем сегментированный кадр
                self.show_segmentation = True
                self.display_frame()
                self.status_var.set("Сегментация выполнена")
            else:
                messagebox.showinfo("Информация", "Не удалось выполнить сегментацию")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при выполнении сегментации: {e}")
            import traceback
            traceback.print_exc()  # Печатаем полный стек ошибки
    
    def toggle_segmentation(self):
        """Переключение режима отображения между обычным кадром и сегментированным"""
        if self.segmented_frame is None:
            messagebox.showinfo("Информация", "Сначала выполните сегментацию")
            return
            
        self.show_segmentation = not self.show_segmentation
        self.display_frame()
        
        if self.show_segmentation:
            self.status_var.set("Отображение сегментации")
        else:
            self.status_var.set("Отображение обычного кадра")
    
    def save_segmentation(self):
        """Сохранение результатов сегментации (маски и bounding box) в файл"""
        if not hasattr(self, 'segmentation_result') or self.segmentation_result is None:
            messagebox.showinfo("Информация", "Сначала выполните сегментацию")
            return
        
        # Запрашиваем имя файла для сохранения
        output_dir = filedialog.askdirectory(title="Выберите директорию для сохранения")
        if not output_dir:
            return

        try:
            # Базовое имя файла (без расширения)
            base_filename = os.path.join(output_dir, f"frame_{self.current_frame_idx + 1}_seg")
            
            # 1. Сохраняем исходный кадр с наложенной маской
            #cv2.imwrite(f"{base_filename}_mask.png", self.segmented_frame)
            
            # 2. Сохраняем только маску (если она доступна)
            if hasattr(self.segmentation_result, 'masks') and self.segmentation_result.masks is not None:
                masks = self.segmentation_result.masks.data
                if len(masks) > 0:
                    # Преобразуем маску в изображение (255 для пикселей маски, 0 для фона)
                    mask_img = (masks[0].cpu().numpy() * 255).astype(np.uint8)
                    #cv2.imwrite(f"{base_filename}_binary_mask.png", mask_img)
            
            # 3. Сохраняем bounding boxes в JSON
            if hasattr(self.segmentation_result, 'boxes') and self.segmentation_result.boxes is not None:
                boxes = self.segmentation_result.boxes.data
                if len(boxes) > 0:
                    # Получаем данные bounding box
                    bbox_data = []
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2, conf, cls = box.tolist()
                        bbox_data.append({
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": conf,
                            "class": int(cls)
                        })
                    
                    # Также добавляем точки, которые были использованы для сегментации
                    points_data = {str(label): {"x": int(x), "y": int(y)} for label, (x, y) in self.points.items()}
                    
                    # Создаем итоговые данные
                    output_data = {
                        "frame_index": self.current_frame_idx,
                        "frame_number": self.current_frame_idx + 1,
                        "video_path": self.video_path,
                        "points": points_data,
                        "bounding_boxes": bbox_data
                    }
                    
                    # Сохраняем в JSON
                    #with open(f"{base_filename}_data.json", 'w', encoding='utf-8') as f:
                    #    json.dump(output_data, f, indent=4)

                    
                    # 4. Сохраняем в формате YOLOv8
                    # Получаем размеры изображения
                    img_height, img_width = self.frame.shape[:2]
                    
                    # Создаем файл с аннотациями в формате YOLOv8
                    with open(f"{base_filename}.txt", 'w', encoding='utf-8') as f:
                        for box in bbox_data:
                            # Преобразуем координаты из формата [x1, y1, x2, y2] в формат YOLO [x_center, y_center, width, height]
                            x1, y1, x2, y2 = box["bbox"]
                            
                            # Вычисляем центр и размеры
                            x_center = (x1 + x2) / (2 * img_width)
                            y_center = (y1 + y2) / (2 * img_height)
                            width = (x2 - x1) / img_width
                            height = (y2 - y1) / img_height
                            
                            # Записываем в формате: <class> <x_center> <y_center> <width> <height> <confidence>
                            f.write(f"{box['class']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            messagebox.showinfo("Успешно", f"Результаты сегментации сохранены в:\n{base_filename}_*")
            self.status_var.set(f"Результаты сегментации сохранены: {os.path.basename(base_filename)}_*")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить результаты сегментации:\n{e}")
            import traceback
            traceback.print_exc()

def main():
    root = tk.Tk()
    app = TkinterVideoAnnotator(root)
    
    # Привязываем обработчик изменения размера
    root.bind("<Configure>", app.on_resize)
    
    # Привязываем обработчик закрытия окна
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Определяем горячие клавиши
    root.bind("<Right>", lambda event: app.next_frame())
    root.bind("<Left>", lambda event: app.prev_frame())
    root.bind("<Control-s>", lambda event: app.save_points())
    root.bind("<Control-o>", lambda event: app.load_video())
    
    # Запускаем приложение
    root.mainloop()

if __name__ == "__main__":
    main()