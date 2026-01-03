#!/usr/bin/env python3
"""YOLO11s to TensorRT FP16 Export - Jetson Orin NX"""

from ultralytics import YOLO

MODEL_PATH = "model/yolo11s_best.pt"

model = YOLO(MODEL_PATH)

model.export(
    format="engine",
    half=True,        # FP16 precision
    batch=1,          
    imgsz=640,
    simplify=False,
)