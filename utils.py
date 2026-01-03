import cv2
import threading
import numpy as np
from typing import Optional


class GStreamerCamera:
    """Hardware-accelerated camera stream using GStreamer"""
    
    def __init__(self, device: int = 0, width: int = 640, height: int = 480, 
                 fps: int = 30, use_hardware: bool = True):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.use_hardware = use_hardware
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def _build_pipelines(self) -> list:
        pipelines = []
        if self.use_hardware:
            pipelines.append(
                f"v4l2src device=/dev/video{self.device} ! "
                "video/x-raw,format=YUY2 ! "
                "nvvidconv ! video/x-raw,format=BGRx ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=true sync=false max-buffers=1"
            )
            pipelines.append(
                f"v4l2src device=/dev/video{self.device} ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=true sync=false max-buffers=1"
            )
        return pipelines
    
    def _connect(self) -> bool:
        for pipeline in self._build_pipelines():
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return True
                self.cap.release()
        
        self.cap = cv2.VideoCapture(self.device)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return True
        
        return False
    
    def _read_loop(self):
        while self._running:
            if self.cap is None or not self.cap.isOpened():
                break
            
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._lock:
                    self.frame = frame
    
    def start(self) -> bool:
        if not self._connect():
            return False
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return True
    
    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self.frame is not None:
                return True, self.frame.copy()
            return False, None
