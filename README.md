# YOLOs-CPP



&#x20;  &#x20;

## Overview

**YOLOs-CPP** is a high-performance C++ library for real-time object detection, segmentation, oriented object detection (OBB), and pose estimation using multiple YOLO model versions. It integrates ONNX Runtime and OpenCV to support fast, flexible inference across a variety of input types (image, video, camera).

## Features

- **Multiple YOLO Models**: Support for YOLOv5 to YOLOv12
- **Detection Types**: Standard detection, segmentation, OBB, and pose estimation
- **Backends**: ONNX Runtime for GPU/CPU acceleration
- **Real-Time**: Optimized for real-time performance
- **Cross-Platform**: Linux, Windows, macOS
- **Easy Integration**: Modular headers and examples for C++ projects

## 🔄 Recent Updates

- **[2025.05.15]**: Classification support added
- **[2025.03.16]**: Pose estimation support
- **[2025.02.11]**: OBB support
- **[2025.01.29]**: YOLOv9+ support
- **[2024.10.23]**: Initial release

## Quick Start

### Clone the Repository

```bash
git clone https://github.com/Geekgineer/YOLOs-CPP
cd YOLOs-CPP
```

### Build

```bash
./build.sh
```

### Run Inference

- Image: `./run_image.sh`
- Video: `./run_video.sh`
- Camera: `./run_camera.sh`

## Supported Models

| Type         | Examples                   |
| ------------ | -------------------------- |
| Standard     | yolo11n.onnx, yolo12n.onnx |
| Segmentation | yolo11n-seg.onnx           |
| OBB          | yolo11n-obb.onnx           |
| Pose         | yolo11n-pose.onnx          |
| Quantized    | yolo11n\_uint8.onnx        |

Custom ONNX export recommended via `models/export_onnx.py`.

## Demo

> For full installation, usage, contribution, and model details, see the `docs/` folder.

---

Licensed under the MIT License. See `docs/LICENSE.md`.

### Acknowledgments

See `docs/ACKNOWLEDGMENTS.md` for external contributions and references.


