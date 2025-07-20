# YOLOs-CPP - Real-Time Object Detection with YOLO Models in C++

![Cover Image](data/cover.png)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![C++](https://img.shields.io/badge/language-C++-blue.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-v1.19.2-brightgreen.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5.5-brightgreen.svg)
![CMake](https://img.shields.io/badge/CMake-3.22.1-blue.svg)

## Table of Contents
- [Overview](#overview)
- [Latest Updates](#-latest-updates-)
- [Features](#features)
- [Quick Start](#quick-start)
  - [Detection Example](#detection-example)
  - [Segmentation Example](#segmentation-example)
  - [Oriented Detection Example](#oriented-detection-example)
  - [Pose Estimation Example](#pose-estimation-example)
- [Installation](#installation)
- [Usage](#usage)
- [Model Support](#model-support)
- [Quantization](#quantization)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

**YOLOs-CPP** is a high-performance C++ library for real-time object detection, segmentation, oriented object detection (OBB), and pose estimation using various YOLO models from Ultralytics. Built with ONNX Runtime and OpenCV, it provides seamless integration for YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLOv11, and YOLOv12 models.

Key advantages:
- Single-header implementation for easy integration
- Support for image, video, and camera inference
- Cross-platform compatibility (Linux, Windows, macOS)
- Optimized for both CPU and GPU execution

## 📰 Latest Updates 📌

* **[2025.05.15]** Added support for classification tasks
* **[2025.04.04]** Launched [Depths-CPP](https://github.com/Geekgineer/Depths-CPP) for depth estimation
* **[2025.03.16]** Added pose estimation support
* **[2025.02.11]** Added OBB (Oriented Bounding Box) format support
* **[2025.02.19]** Added YOLOv12 support
* **[2025.01.29]** Added YOLOv9 support
* **[2025.01.26]** Added segmentation headers for YOLOv9
* **[2025.01.26]** Added segmentation support for YOLOv8 and YOLOv11 with quantized models
* **[2024.10.23]** Initial project launch with detection support

## Features

- **Multiple YOLO Model Support**: v5, v7, v8, v9, v10, v11, v12
- **Multiple Task Support**: Detection, Segmentation, OBB, Pose Estimation
- **ONNX Runtime Integration**: Optimized inference on CPU/GPU
- **Dynamic Shapes Handling**: Adapts to varying input sizes
- **Efficient Processing**: NMS, Rotated NMS, and keypoint-based filtering
- **Real-Time Performance**: For images, videos, and camera streams
- **Cross-Platform**: Works on Linux, Windows, and macOS
- **Quantization Support**: Reduced model size with minimal accuracy trade-off

## Quick Start

### Detection Example

```cpp
#include <opencv2/opencv.hpp>
#include "det/YOLO11.hpp"

int main() {
    // Configuration
    const std::string labelsPath = "../models/coco.names";
    const std::string modelPath = "../models/yolo11n.onnx";
    const std::string imagePath = "../data/dogs.jpg";
    bool isGPU = true;

    // Initialize detector
    YOLO11Detector detector(modelPath, labelsPath, isGPU);
    
    // Load image and detect
    cv::Mat image = cv::imread(imagePath);
    std::vector<Detection> detections = detector.detect(image);
    
    // Draw results and display
    detector.drawBoundingBoxMask(image, detections);
    cv::imshow("Detections", image);
    cv::waitKey(0);
    
    return 0;
}

#include <opencv2/opencv.hpp>
#include "seg/YOLO11Seg.hpp"

int main() {
    // Configuration
    const std::string labelsPath = "../models/coco.names";
    const std::string modelPath = "../models/yolo11n-seg.onnx";
    const std::string imagePath = "../data/dogs.jpg";
    bool isGPU = true;

    // Initialize segmentor
    YOLOv11SegDetector segmentor(modelPath, labelsPath, isGPU);
    
    // Load image and segment
    cv::Mat image = cv::imread(imagePath);
    std::vector<Segmentation> results = segmentor.segment(image, 0.2f, 0.45f);
    
    // Draw results and display
    segmentor.drawSegmentations(image, results);
    cv::imshow("Segmentation", image);
    cv::waitKey(0);
    
    return 0;
}

#include <opencv2/opencv.hpp>
#include "obb/YOLO11-OBB.hpp"

int main() {
    // Configuration
    const std::string labelsPath = "../models/Dota.names";
    const std::string modelPath = "../models/yolo11n-obb.onnx";
    const std::string imagePath = "../data/frame_37.jpg";
    bool isGPU = true;

    // Initialize detector
    YOLO11OBBDetector detector(modelPath, labelsPath, isGPU);
    
    // Load image and detect
    cv::Mat image = cv::imread(imagePath);
    std::vector<Detection> results = detector.detect(image);
    
    // Draw results and display
    detector.drawBoundingBox(image, results);
    cv::imshow("OBB Detection", image);
    cv::waitKey(0);
    
    return 0;
}

#include <opencv2/opencv.hpp>
#include "pose/YOLO11-POSE.hpp"

int main() {
    // Configuration
    const std::string labelsPath = "../models/coco.names";
    const std::string modelPath = "../models/yolo11n-pose.onnx";
    const std::string imagePath = "../data/person.jpg";
    bool isGPU = true;

    // Initialize pose detector
    YOLO11POSEDetector poseDetector(modelPath, labelsPath, isGPU);
    
    // Load image and estimate poses
    cv::Mat image = cv::imread(imagePath);
    std::vector<PoseDetection> poses = poseDetector.detect(image);
    
    // Draw results and display
    poseDetector.drawBoundingBox(image, poses);
    cv::imshow("Pose Estimation", image);
    cv::waitKey(0);
    
    return 0;
}

git clone https://github.com/Geekgineer/YOLOs-CPP
cd YOLOs-CPP

# Configure build (edit build.sh for ONNX Runtime version)
./build.sh

# Run examples
./run_image.sh    # Image inference
./run_video.sh    # Video inference
./run_camera.sh   # Camera inference
