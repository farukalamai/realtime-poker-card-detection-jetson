# Real-Time Poker Card Detection on Jetson Orin NX

Real-time 52-class playing card detection using YOLO11s.

## Environment

**Jetson Orin NX 16GB**
- JetPack 6.2 (L4T R36.4.7)
- CUDA 12.6 | TensorRT 10.7 | cuDNN 9.17
- Python 3.10 | PyTorch 2.8

## Setup

### TensorRT Export

```bash
python scripts/yolo_to_fp16.py
```

## Usage

```bash
python main.py
```

## Training

Dataset: [Playing Cards](https://universe.roboflow.com/augmented-startups/playing-cards-ow27d/dataset/4) (Roboflow)




