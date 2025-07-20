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
