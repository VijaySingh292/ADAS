# ADAS and Traffic Prediction System

This repository contains an integrated system for real-time traffic analysis, road condition monitoring, and vehicle flow prediction using deep learning models. It leverages multiple AI models including YOLO for object detection, EfficientNet/ConvNeXt for road surface and weather classification, and an LSTM ensemble with attention mechanisms for traffic prediction.

## Features
- **Object Detection**: Detects vehicles, obstacles, and road signs from dashcam and CCTV footage using YOLO models.
- **Road Surface Classification**: Classifies road conditions (e.g., dry asphalt, wet mud, ice) using a ConvNeXt model.
- **Weather Classification**: Identifies weather conditions (Clear, Fog, Rain) using EfficientNet.
- **Traffic Prediction**: Forecasts vehicle counts on left and right lanes using an LSTM ensemble with red light attention.
- **Real-Time Processing**: Processes alternating dashcam and CCTV video streams with driving recommendations.
- **Risk Assessment**: Calculates a risk score based on weather, surface conditions, and detected objects.
- **Visualization**: Generates training metrics plots and annotated video outputs.

## Repository Structure
- `object.ipynb`: Trains a YOLOv8 model for object detection.
- `surface.ipynb`: Trains an EfficientNet-based model for road surface classification.
- `traffic_prediction.ipynb`: Trains an LSTM ensemble for traffic prediction with red light emphasis.
- `integration.ipynb`: Combines all models for real-time video processing and recommendations.
- `road_seg.ipynb` : For road segmentation
- `weather.ipynb` : For training a weather detection model 

## Prerequisites
- **Hardware**: 
  - GPU recommended (NVIDIA CUDA-compatible for PyTorch and TensorFlow).
  - Minimum 16GB RAM for training and inference.
- **Software**:
  - Python 3.8+
  - CUDA 11.x (if using GPU)
  - Dependencies listed in `requirements.txt`

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/VijaySingh292/ADAS_Project.git
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Real-Time Video Processing
Run the integrated system on dashcam and CCTV videos:
```bash
python integration.ipynb
```
- **Inputs**: 
  - Dashcam video: `path/to/dashcam.mp4`
  - CCTV video: `path/to/cctv.mp4`
  - Timestamp: e.g., `"2025-04-03 15:03:00"`
  - Day: e.g., `"Thursday"`
  - Red light status: `"yes"` or `"no"`
- **Outputs**: Annotated videos (`dashcam_output.mp4`, `cctv_output.mp4`) with predictions and recommendations.


## Model Details
- **YOLOv8 (Object Detection)**:
  - Detects: animals, vehicles, people, speedbumps, obstacles, signs, traffic lights.
  - Trained on custom dataset with 8 classes.
- **YOLO (Road Segmentation)**:
  - Segments road areas for precise object localization.
- **EfficientNet-B0 (Weather)**:
  - Classes: Clear, Fog, Rain.
- **ConvNeXt-Small (Surface)**:
  - 25+ classes grouped into "Easy to Drive", "Take Precautions", "Dangerous".
- **LSTM Ensemble (Traffic)**:
  - 3 models with red light attention mechanism.
  - Predicts vehicle counts for left and right lanes.

## Output Examples
- **Videos**: Annotated with detected objects, weather, surface conditions, risk scores, and recommendations.
- **Plots**: Training loss, RMSE, prediction vs. actual comparisons, error distributions.

## Training Data
- **Object Detection**: Custom dataset with images and annotations(with 8 classes).
- **Road Detection**: Yolo Dataset from roboflow 
- **Weather Detection**: Three folder labelled as CLear , Fog and Rain 
- **Road Surface**: Balanced RSCD dataset with train/val/test with 27 classes
- **Traffic Prediction**: CSV files with timestamp, day, red light status, and vehicle counts(Its is a synthetic data and i have given the code in traffic_prediction.ipynb to generate it)
- 
## Results 

![Dashcam Output](images/Screenshot%202025-04-09%20085949.png)
![CCTV Output](images/Screenshot%202025-04-09%20094609.png)

## Acknowledgments
- Built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), and [timm](https://github.com/rwightman/pytorch-image-models).
- Inspired by real-world traffic monitoring systems and Greenshields traffic flow model.

---

### Notes for Customization
- Update file paths to match your environment (e.g., Google Drive paths in the original code).

