from ultralytics import YOLO
import cv2
import time  # Add this import

# Load a model
model = YOLO("yolo11n.pt")

# Export the model to NCNN format
model.export(format="ncnn", half=True)

# Initialize webcam
#cap = cv2.VideoCapture(0)  # 0 is usually the default USB camera

# Установите разрешение
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# Initialize variables for FPS calculation
target_fps = 8
frame_interval = 1.0 / target_fps  # Интервал между обработкой кадров в секундах

last_time_processed = time.time()
mean_fps = []
while cap.isOpened():
    # Read frame
    success, frame = cap.read()
    
    if success:
        # Calculate FPS
        current_time = time.time()

        # Проверяем, прошло ли достаточно времени для обработки следующего кадра
        if current_time - last_time_processed >= frame_interval:
            # Обновляем время последней обработки
            last_time_processed = current_time
            # Run YOLOv8 inference on the frame
            results = model(frame, half=True, device=0)
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
            curr_time = time.time()
            fps = 1/(curr_time - last_time_processed)
            mean_fps.append(fps)
        if len(mean_fps) > 50:
            fps = (sum(mean_fps[-50:]) / 50) / 8
        else:
            fps = (sum(mean_fps) / len(mean_fps)) / 8
        # Add FPS text to the frame
        cv2.putText(annotated_frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, (0, 255, 0), 2)
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the capture object and close windows
cap.release()
cv2.destroyAllWindows()