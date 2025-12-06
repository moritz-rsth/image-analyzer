# Replicate Models

This directory contains the Replicate model definitions for deploying machine learning models using [Cog](https://github.com/replicate/cog). These models are designed to be deployed on [Replicate](https://replicate.com) for scalable, cloud-based inference.

## Overview

This repository includes three main models:

1. **detect_faces** - Face detection using MTCNN (Multi-task Cascaded Convolutional Networks)
2. **predict_coco_labels_yolo11** - Object detection and classification using YOLO11 with COCO labels
3. **predict_imagenet_classes_yolo11** - Image classification using YOLO11 with ImageNet classes

## Directory Structure

Each model directory contains:
- `cog.yaml` - Cog configuration file defining the model build and runtime settings
- `predict.py` - Python prediction script with the model inference logic
- `requirements.txt` - Python dependencies required for the model
- `README.md` - Model-specific documentation

## Prerequisites

Before deploying these models, ensure you have:

1. **Cog installed**: Install Cog following the [official documentation](https://github.com/replicate/cog#install)
   ```bash
   sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
   sudo chmod +x /usr/local/bin/cog
   ```

2. **Replicate account**: Sign up at [replicate.com](https://replicate.com) and authenticate:
   ```bash
   cog login
   ```

3. **Docker**: Cog requires Docker to be installed and running

## Deployment Instructions

### General Deployment Process

For each model, follow these steps:

1. **Navigate to the model directory**:
   ```bash
   cd replicate_models/<model_name>
   ```

2. **Build the model image**:
   ```bash
   cog build -t <your-username>/<model-name>
   ```

3. **Push to Replicate**:
   ```bash
   cog push <your-username>/<model-name>
   ```

4. **Create a deployment on Replicate**:
   - Visit [replicate.com/create](https://replicate.com/create)
   - Select your pushed model
   - Configure hardware settings (GPU recommended for all models)
   - Set up API access and billing

### Model-Specific Details

#### detect_faces
- **Model**: MTCNN for face detection
- **Hardware**: GPU recommended (Nvidia T4 or better), CPU supported but slower
- **Input**: Image file, optional confidence threshold
- **Output**: JSON with face count, bounding boxes, landmarks, and confidence scores

#### predict_coco_labels_yolo11
- **Model**: YOLO11 with COCO dataset labels
- **Hardware**: GPU required for optimal performance
- **Input**: Image file
- **Output**: Detected objects with COCO class labels, bounding boxes, and confidence scores

#### predict_imagenet_classes_yolo11
- **Model**: YOLO11 with ImageNet class labels
- **Hardware**: GPU required for optimal performance
- **Input**: Image file
- **Output**: Image classification results with ImageNet class labels and confidence scores

## Configuration Files

### cog.yaml
Each model's `cog.yaml` file defines:
- Build requirements (Python version, system packages)
- GPU/CPU requirements
- Runtime configuration
- Model image name and prediction entry point

### predict.py
Contains the `Predictor` class that:
- Loads the model weights (downloaded automatically on first run)
- Processes input images
- Returns predictions in the expected format

### requirements.txt
Lists all Python package dependencies needed for the model to run.

## Model Weights

Model weights (`.pt`, `.pth`, `.weights`, `.h5`, `.ckpt`, `.pkl` files) are **not** included in this repository. They are automatically downloaded when the model is first run or built. This keeps the repository size manageable and ensures you're using the latest compatible weights.

## Troubleshooting

### Common Issues

1. **Build failures**: Ensure all dependencies in `requirements.txt` are compatible and available
2. **GPU out of memory**: Reduce batch size or use a smaller model variant
3. **Authentication errors**: Run `cog login` again to refresh credentials
4. **Docker issues**: Ensure Docker is running and you have sufficient disk space

### Getting Help

If you encounter issues or have questions about deploying these models:
- Check the model-specific README.md in each subdirectory
- Review the [Cog documentation](https://github.com/replicate/cog)
- Review the [Replicate documentation](https://replicate.com/docs)

## Contact

For questions, issues, or contributions, please contact:
**moritz.rosenthal@tum.de**

## License

Please refer to the main repository license for usage terms.

