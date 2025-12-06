import cv2
import json
import logging
import numpy as np
import os
import sys
import torch  # type: ignore  # Only available in Replicate environment
from cog import BasePredictor, Input, Path  # type: ignore  # Only available in Replicate environment
from mtcnn import MTCNN

# Setup logging to stdout/stderr for Replicate monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


class Predictor(BasePredictor):
    def setup(self):
        """Load the MTCNN face detector model"""
        logger.info("Starting MTCNN face detector setup...")
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            logger.info("Initializing MTCNN detector...")
            # MTCNN automatically uses the best available hardware
            # The device parameter is not supported in this version
            self.detector = MTCNN()
            logger.info("MTCNN detector initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize MTCNN detector: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
    def predict(
        self,
        image: Path = Input(description="Image to detect faces in"),
        confidence_threshold: float = Input(
            description="Confidence threshold for face detection (0.0-1.0)",
            default=0.85,
            ge=0.0,
            le=1.0
        )
    ) -> str:
        """
        Detect faces in an image using MTCNN.
        
        Returns a JSON string with face detection results including:
        - face_count: Number of faces detected
        - face_confidence_avg: Average confidence score
        - face_area_total_abs: Total absolute face area in pixels
        - face_area_total_rel: Total relative face area (0-1)
        - faces: List of detected faces with bounding boxes and landmarks
        """
        try:
            logger.info(f"Starting face detection with confidence_threshold={confidence_threshold}")
            logger.info(f"Image path: {image}")
            
            # Read image
            logger.info("Reading image...")
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                error_msg = f"Could not read image: {image}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            height, width = img_bgr.shape[:2]
            logger.info(f"Image read successfully. Shape: {img_bgr.shape} (height={height}, width={width})")
            
            # Convert BGR to RGB for MTCNN
            logger.info("Converting BGR to RGB...")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            logger.info("Running MTCNN face detection...")
            # MTCNN detect_faces() doesn't support box_format or threshold_onet parameters
            # It returns boxes in [x, y, width, height] format by default
            faces = self.detector.detect_faces(img_rgb)
            logger.info(f"Face detection completed. Found {len(faces) if faces else 0} faces")
            
            # Filter faces by confidence threshold
            if faces:
                filtered_faces = [
                    face for face in faces 
                    if face.get("confidence", 0.0) >= confidence_threshold
                ]
                faces = filtered_faces
                logger.info(f"After filtering (confidence >= {confidence_threshold}): {len(faces)} faces")
            
            # Process results
            if not faces:
                logger.info("No faces detected in image")
                return json.dumps({
                    "face_count": 0,
                    "face_confidence_avg": None,
                    "face_area_total_abs": 0,
                    "face_area_total_rel": 0.0,
                    "faces": []
                })
            
            # Extract face information
            logger.info(f"Processing {len(faces)} detected faces...")
            face_list = []
            scores = []
            areas_abs = []
            areas_rel = []
            
            for idx, face in enumerate(faces):
                try:
                    confidence = face["confidence"]
                    box = face["box"]  # [x, y, width, height]
                    landmarks = face["keypoints"]
                    
                    # Validate required keys exist
                    if "confidence" not in face or "box" not in face or "keypoints" not in face:
                        logger.warning(f"Face {idx+1} missing required keys, skipping")
                        continue
                    
                    face_info = {
                        "face_id": idx + 1,
                        "confidence": float(confidence),
                        "bounding_box": {
                            "x": int(box[0]),
                            "y": int(box[1]),
                            "width": int(box[2]),
                            "height": int(box[3])
                        },
                        "landmarks": {
                            "right_eye": [int(landmarks["right_eye"][0]), int(landmarks["right_eye"][1])],
                            "left_eye": [int(landmarks["left_eye"][0]), int(landmarks["left_eye"][1])],
                            "nose": [int(landmarks["nose"][0]), int(landmarks["nose"][1])],
                            "mouth_right": [int(landmarks["mouth_right"][0]), int(landmarks["mouth_right"][1])],
                            "mouth_left": [int(landmarks["mouth_left"][0]), int(landmarks["mouth_left"][1])]
                        }
                    }
                    face_list.append(face_info)
                    
                    scores.append(confidence)
                    area_abs = box[2] * box[3]
                    areas_abs.append(area_abs)
                    areas_rel.append(area_abs / (height * width))
                    
                    # Safe formatting - ensure confidence is a number before formatting
                    try:
                        conf_val = float(confidence) if confidence is not None else 0.0
                        logger.debug(f"Face {idx+1}: confidence={conf_val:.3f}, area={area_abs}px")
                    except (ValueError, TypeError):
                        logger.debug(f"Face {idx+1}: confidence={confidence}, area={area_abs}px")
                except Exception as e:
                    logger.warning(f"Error processing face {idx+1}: {str(e)}")
                    continue
            
            logger.info(f"Successfully processed {len(face_list)} faces")
            
            # Calculate aggregate statistics
            avg_confidence = sum(scores) / len(scores) if scores else None
            total_area_abs = sum(areas_abs)
            total_area_rel = sum(areas_rel)
            
            # Safe formatting for all statistics
            avg_conf_str = f"{avg_confidence:.3f}" if avg_confidence is not None else "None"
            try:
                total_area_rel_str = f"{total_area_rel:.3f}"
            except (ValueError, TypeError):
                total_area_rel_str = str(total_area_rel)
            logger.info(f"Statistics: avg_confidence={avg_conf_str}, total_area_abs={total_area_abs}, total_area_rel={total_area_rel_str}")
            
            # Build result dictionary with safe type conversions
            try:
                result = {
                    "face_count": len(faces),
                    "face_confidence_avg": float(avg_confidence) if avg_confidence is not None else None,
                    "face_area_total_abs": int(total_area_abs) if total_area_abs is not None else 0,
                    "face_area_total_rel": float(min(1.0, total_area_rel)) if total_area_rel is not None else 0.0,
                    "faces": face_list
                }
                
                logger.info("Face detection completed successfully")
                return json.dumps(result, indent=2)
            except Exception as result_error:
                # Fallback: return result even if JSON serialization fails
                logger.error(f"Error creating result JSON: {str(result_error)}")
                # Try to return a simpler result
                try:
                    simple_result = {
                        "face_count": len(faces) if faces else 0,
                        "face_confidence_avg": None,
                        "face_area_total_abs": 0,
                        "face_area_total_rel": 0.0,
                        "faces": []
                    }
                    return json.dumps(simple_result, indent=2)
                except Exception:
                    # Last resort: return minimal JSON
                    return json.dumps({"face_count": 0, "face_confidence_avg": None, "face_area_total_abs": 0, "face_area_total_rel": 0.0, "faces": []})
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            logger.error(f"Error during face detection: {error_msg}")
            logger.error(f"Traceback: {error_traceback}")
            error_result = {
                "error": error_msg,
                "traceback": error_traceback,
                "face_count": 0,
                "face_confidence_avg": None,
                "face_area_total_abs": 0,
                "face_area_total_rel": 0.0,
                "faces": []
            }
            return json.dumps(error_result)

