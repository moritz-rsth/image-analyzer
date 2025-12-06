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
        """Load the YOLO11 classification model"""
        logger.info("Starting model setup...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Download model weights if not present
        model_path = "yolo11n-cls.pt"
        if not os.path.exists(model_path):
            logger.info("Downloading YOLO11 classification model weights...")
            import urllib.request
            url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt"
            urllib.request.urlretrieve(url, model_path)
            logger.info("Model weights downloaded successfully")
        else:
            logger.info(f"Model weights found at {model_path}")
        
        # Load model
        logger.info("Loading YOLO model...")
        self.model = YOLO(model_path).to(self.device)
        logger.info("Model loaded successfully")
        
    def predict(
        self,
        image: Path = Input(description="Image to classify"),
        top_k: int = Input(
            description="Number of top predictions to return",
            default=10,
            ge=1,
            le=1000
        )
    ) -> str:
        """
        Predict ImageNet classes in an image using YOLO11 classification model.
        
        Returns a JSON string with classification results including:
        - predictions: List of top-k predictions with class names and probabilities
        - all_classes: Dictionary with all ImageNet classes and their probabilities
        """
        try:
            logger.info(f"Starting prediction with top_k={top_k}")
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
            results = self.model(img, verbose=False)
            logger.info(f"Inference completed. Results type: {type(results)}")
            
            if not results:
                error_msg = "No results from model"
                logger.error(error_msg)
                return json.dumps({
                    "error": error_msg,
                    "predictions": [],
                    "all_classes": {}
                })
            
            # Process results
            logger.info("Processing results...")
            result = results[0]
            logger.info(f"Result type: {type(result)}")
            
            # Check if probs attribute exists
            if not hasattr(result, 'probs'):
                error_msg = f"Result object does not have 'probs' attribute. Available attributes: {dir(result)}"
                logger.error(error_msg)
                raise AttributeError(error_msg)
            
            probs = result.probs
            logger.info(f"Probs type: {type(probs)}")
            logger.info(f"Probs attributes: {[attr for attr in dir(probs) if not attr.startswith('_')]}")
            
            # Get probabilities as tensor/array
            # The Probs object has a 'data' attribute according to the error message
            # Check if 'data' attribute exists
            if not hasattr(probs, 'data'):
                error_msg = f"Probs object does not have 'data' attribute. Available attributes: {[attr for attr in dir(probs) if not attr.startswith('_')]}"
                logger.error(error_msg)
                raise AttributeError(error_msg)
            
            # Use the 'data' attribute to get all probabilities
            # This is the correct way according to the Probs class documentation
            prob_data = probs.data  # This is a tensor/array with all probabilities
            logger.info(f"Prob data type: {type(prob_data)}, shape: {prob_data.shape if hasattr(prob_data, 'shape') else 'N/A'}")
            
            # Convert to numpy if it's a torch tensor
            if torch.is_tensor(prob_data):
                logger.info("Converting torch tensor to numpy...")
                prob_array = prob_data.cpu().numpy()
            else:
                # Ensure it's a numpy array (in case it's already numpy or another type)
                logger.info("Converting to numpy array...")
                prob_array = np.asarray(prob_data)
            
            logger.info(f"Prob array shape: {prob_array.shape}, dtype: {prob_array.dtype}")
            
            # Get top-k predictions manually using numpy argsort
            # argsort returns indices sorted by value (ascending), so we reverse it
            # NOTE: We do NOT use probs.topk() as it doesn't exist - we use probs.data instead
            logger.info(f"Computing top-{top_k} predictions using numpy argsort...")
            top_k_indices = np.argsort(prob_array)[::-1][:top_k]
            top_k_probs = prob_array[top_k_indices]
            logger.info(f"Top-k indices: {top_k_indices[:5]}... (showing first 5)")
            
            # Check if names attribute exists
            if not hasattr(result, 'names'):
                error_msg = f"Result object does not have 'names' attribute. Available attributes: {dir(result)}"
                logger.error(error_msg)
                raise AttributeError(error_msg)
            
            predictions = []
            for idx, prob in zip(top_k_indices, top_k_probs):
                class_name = result.names[idx]
                predictions.append({
                    "class": class_name,
                    "probability": float(prob),
                    "class_id": int(idx)
                })
            logger.info(f"Created {len(predictions)} predictions")
            
            # Get all class probabilities
            logger.info("Processing all class probabilities...")
            all_classes = {}
            if torch.is_tensor(prob_data):
                prob_list = prob_data.cpu().tolist()
            else:
                prob_list = prob_data.tolist()
            
            for i, prob in enumerate(prob_list):
                class_name = result.names[i]
                all_classes[f"imagenet_{class_name}"] = float(prob)
            
            logger.info(f"Processed {len(all_classes)} total classes")
            
            result_dict = {
                "predictions": predictions,
                "all_classes": all_classes
            }
            
            logger.info("Prediction completed successfully")
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
                "predictions": [],
                "all_classes": {}
            }
            return json.dumps(error_result)

