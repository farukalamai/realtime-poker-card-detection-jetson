import cv2
import time
import argparse
import os
from ultralytics import YOLO
from utils import GStreamerCamera

ENGINE_PATH = "model/yolo11s_best.engine"
CONF_THRESHOLD = 0.25

def main(video_path=None):
    model = YOLO(ENGINE_PATH, task='detect')
    
    video_writer = None
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return
        use_gstreamer = False
        
        # Setup video writer for saving processed video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Generate output path in same folder as input
        base_name = os.path.splitext(video_path)[0]
        output_path = f"{base_name}_processed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Processing video: {video_path}")
        print(f"Output will be saved to: {output_path}")
    else:
        cap = GStreamerCamera(device=0)
        if not cap.start():
            print("Error: Could not open camera")
            return
        use_gstreamer = True
    
    prev_time = time.time()
    frame_count = 0
    skip_frames = 2
    annotated_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            if video_path:
                break
            continue
        
        frame_count += 1
        if frame_count % (skip_frames + 1) == 0:
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            annotated_frame = results[0].plot(conf=False)
        
        if annotated_frame is None:
            annotated_frame = frame
        
        if video_path:
            # Calculate FPS
            curr_time = time.time()
            fps_val = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            # Make a copy to avoid drawing FPS multiple times on same frame
            output_frame = annotated_frame.copy()
            
            # Add FPS to the frame before saving (fixed position)
            fps_text = f"FPS: {fps_val:.1f}"
            text_x = output_frame.shape[1] - 150
            cv2.putText(output_frame, fps_text, (text_x, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save frame to output video (with FPS overlay)
            video_writer.write(output_frame)
            
            # Display the processed frame
            h, w = output_frame.shape[:2]
            display_frame = cv2.resize(output_frame, (int(w * 0.85), int(h * 0.85)))
            
            cv2.imshow("Processing Video", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nProcessing interrupted by user.")
                break
            
            # Print progress
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
        else:
            # Resize to 85% of original for display
            h, w = annotated_frame.shape[:2]
            display_frame = cv2.resize(annotated_frame, (int(w * 0.85), int(h * 0.85)))
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            # Display FPS in top right corner (fixed position)
            fps_text = f"FPS: {fps:.1f}"
            text_x = display_frame.shape[1] - 150
            cv2.putText(display_frame, fps_text, (text_x, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Poker Card Detection", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    if use_gstreamer:
        cap.stop()
    else:
        cap.release()
    
    if video_writer:
        video_writer.release()
        print(f"\nProcessing complete! Saved to: {output_path}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poker Card Detection")
    parser.add_argument("--video", "-v", type=str, default=None,
                        help="Path to video file. If not provided, uses webcam.")
    args = parser.parse_args()
    main(video_path=args.video)
