import cv2
import json
import logging
import numpy as np
import os
import sys
import torch  # type: ignore  # Only available in Replicate environment
from cog import BasePredictor, Input, Path  # type: ignore  # Only available in Replicate environment
from ultralytics import YOLO  # type: ignore  # Only available in Replicate environment

# Setup logging to stdout/stderr for Replicate monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


class Predictor(BasePredictor):
    def setup(self):
        """Load the YOLO11 detection model and COCO labels"""
        logger.info("Starting model setup...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Download model weights if not present
        model_path = "yolo11n.pt"
        if not os.path.exists(model_path):
            logger.info("Downloading YOLO11 detection model weights...")
            try:
                import urllib.request
                url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
                urllib.request.urlretrieve(url, model_path)
                logger.info(f"Successfully downloaded model to {model_path}")
            except Exception as e:
                error_msg = f"Failed to download model weights: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        else:
            logger.info(f"Model weights found at {model_path}")
        
        # Verify model file exists
        if not os.path.exists(model_path):
            error_msg = f"Model file not found: {model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Load model
        try:
            logger.info(f"Loading YOLO11 model from {model_path}...")
            self.model = YOLO(model_path).to(self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load YOLO11 model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Download COCO labels if not present
        coco_names_path = "coco.names"
        if not os.path.exists(coco_names_path):
            logger.info("Downloading COCO labels...")
            try:
                import urllib.request
                url = "https://opencv-tutorial.readthedocs.io/en/latest/_downloads/a9fb13cbea0745f3d11da9017d1b8467/coco.names"
                
                # Create request with User-Agent header to avoid 403 Forbidden
                # Use the same method as download_weights in helper_functions.py
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
                
                logger.info(f"Downloading from: {url}")
                with urllib.request.urlopen(req) as response:
                    with open(coco_names_path, 'wb') as f:
                        f.write(response.read())
                logger.info(f"Successfully downloaded COCO labels to {coco_names_path}")
            except Exception as e:
                error_msg = f"Failed to download COCO labels: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        else:
            logger.info(f"COCO labels found at {coco_names_path}")
        
        # Verify COCO labels file exists
        if not os.path.exists(coco_names_path):
            error_msg = f"COCO labels file not found: {coco_names_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Load COCO labels
        try:
            logger.info(f"Loading COCO labels from {coco_names_path}...")
            with open(coco_names_path, 'r') as f:
                self.classes = ['coco_' + line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(self.classes)} COCO classes")
        except Exception as e:
            error_msg = f"Failed to load COCO labels: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
    def predict(
        self,
        image: Path = Input(description="Image to detect objects in"),
        confidence_threshold: float = Input(
            description="Confidence threshold for object detection (0.0-1.0)",
            default=0.25,
            ge=0.0,
            le=1.0
        ),
        iou_threshold: float = Input(
            description="IoU threshold for non-maximum suppression (0.0-1.0)",
            default=0.45,
            ge=0.0,
            le=1.0
        )
    ) -> str:
        """
        Detect COCO objects in an image using YOLO11 detection model.
        
        Returns a JSON string with detection results including:
        - detections: List of detected objects with bounding boxes, classes, and confidences
        - class_probabilities: Dictionary with maximum confidence for each COCO class
        - detection_count: Total number of detections
        """
        try:
            logger.info(f"Starting prediction with confidence_threshold={confidence_threshold}, iou_threshold={iou_threshold}")
            logger.info(f"Image path: {image}")
            
            # Read image
            logger.info("Reading image...")
            img = cv2.imread(str(image))
            if img is None:
                error_msg = f"Could not read image: {image}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.info(f"Image read successfully. Shape: {img.shape}")
            
            # Perform inference
            logger.info("Running model inference...")
            results = self.model(
                img, 
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False
            )
            logger.info(f"Inference completed. Results type: {type(results)}")
            
            if not results:
                error_msg = "No results from model"
                logger.error(error_msg)
                return json.dumps({
                    "error": error_msg,
                    "detections": [],
                    "class_probabilities": {},
                    "detection_count": 0
                })
            
            # Process results
            logger.info("Processing results...")
            result = results[0]
            logger.info(f"Result type: {type(result)}")
            
            # Check if boxes attribute exists
            if not hasattr(result, 'boxes'):
                error_msg = f"Result object does not have 'boxes' attribute. Available attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}"
                logger.error(error_msg)
                raise AttributeError(error_msg)
            
            detections = []
            class_max_conf = {cls: 0.0 for cls in self.classes}
            
            if result.boxes is not None:
                logger.info(f"Processing {len(result.boxes)} detections...")
                for idx, box in enumerate(result.boxes):
                    try:
                        class_id = int(box.cls)
                        confidence = float(box.conf)
                        
                        # Validate class_id is within bounds
                        if class_id < 0 or class_id >= len(self.classes):
                            logger.warning(f"Invalid class_id {class_id} (max: {len(self.classes)-1}), skipping detection")
                            continue
                        
                        class_name = self.classes[class_id]
                        
                        # Get bounding box coordinates
                        try:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                        except Exception as e:
                            logger.warning(f"Failed to extract bounding box coordinates for detection {idx}: {str(e)}")
                            continue
                        
                        detection = {
                            "class": class_name,
                            "class_id": class_id,
                            "confidence": confidence,
                            "bounding_box": {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                                "width": float(x2 - x1),
                                "height": float(y2 - y1)
                            }
                        }
                        detections.append(detection)
                        
                        # Update maximum confidence for this class
                        if confidence > class_max_conf[class_name]:
                            class_max_conf[class_name] = confidence
                    except Exception as e:
                        logger.warning(f"Error processing detection {idx}: {str(e)}")
                        continue
            else:
                logger.info("No boxes found in results")
            
            logger.info(f"Processed {len(detections)} valid detections")
            
            result_dict = {
                "detections": detections,
                "class_probabilities": {k: v for k, v in class_max_conf.items() if v > 0.0},
                "detection_count": len(detections)
            }
            
            logger.info(f"Prediction completed successfully. Found {len(detections)} detections across {len([k for k, v in class_max_conf.items() if v > 0.0])} classes")
            return json.dumps(result_dict, indent=2)
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            logger.error(f"Error during prediction: {error_msg}")
            logger.error(f"Traceback: {error_traceback}")
            error_result = {
                "error": error_msg,
                "traceback": error_traceback,
                "detections": [],
                "class_probabilities": {},
                "detection_count": 0
            }
            return json.dumps(error_result)

