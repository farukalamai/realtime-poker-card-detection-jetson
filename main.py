import cv2
import time
from ultralytics import YOLO
from utils import GStreamerCamera

ENGINE_PATH = "model/yolo11s_best.engine"
CONF_THRESHOLD = 0.40

def main():
    model = YOLO(ENGINE_PATH, task='detect')
    cap = GStreamerCamera(device=0)
    
    if not cap.start():
        print("Error: Could not open camera")
        return
    
    prev_time = time.time()
    frame_count = 0
    skip_frames = 2
    annotated_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        
        frame_count += 1
        if frame_count % (skip_frames + 1) == 0:
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            annotated_frame = results[0].plot(conf=False)
        
        if annotated_frame is None:
            annotated_frame = frame
        
        # Resize to 85% of original
        h, w = annotated_frame.shape[:2]
        display_frame = cv2.resize(annotated_frame, (int(w * 0.85), int(h * 0.85)))
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Display FPS in top right corner
        fps_text = f"FPS: {fps:.1f}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = display_frame.shape[1] - text_size[0] - 10
        cv2.putText(display_frame, fps_text, (text_x, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Poker Card Detection", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
