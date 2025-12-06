# COCO Object Detection Model (YOLO11)

Dieses Modell erkennt Objekte in Bildern mit dem YOLO11 Detection Modell und gibt COCO-Labels zurück.

## Hardware-Empfehlung
- **Empfohlen:** GPU (Nvidia T4 oder besser)
- **Minimum:** GPU erforderlich (CPU ist zu langsam)

## Installation und Deployment

### 1. Model auf Replicate erstellen
Besuche https://replicate.com/create und erstelle ein neues Modell mit dem Namen `your-username/predict-coco-labels-yolo11`

### 2. Mit Cog builden und pushen
```bash
cd replicate_models/predict_coco_labels_yolo11
cog build -t your-username/predict-coco-labels-yolo11
cog push your-username/predict-coco-labels-yolo11
```

### 3. Deployment erstellen
Nach dem Push kannst du auf Replicate ein Deployment erstellen und die Hardware auswählen.

## Inputs
- `image` (Path): Bilddatei zur Objekterkennung
- `confidence_threshold` (float, optional): Confidence-Schwelle für Objekterkennung (0.0-1.0, Standard: 0.25)
- `iou_threshold` (float, optional): IoU-Schwelle für Non-Maximum Suppression (0.0-1.0, Standard: 0.45)

## Output
JSON-String mit folgenden Feldern:
- `detections`: Liste der erkannten Objekte mit Bounding Boxes, Klassen und Confidences
- `class_probabilities`: Dictionary mit maximaler Confidence für jede COCO-Klasse
- `detection_count`: Gesamtzahl der Erkennungen

## Beispiel-Output
```json
{
  "detections": [
    {
      "class": "coco_person",
      "class_id": 0,
      "confidence": 0.95,
      "bounding_box": {
        "x1": 100.0,
        "y1": 150.0,
        "x2": 300.0,
        "y2": 500.0,
        "width": 200.0,
        "height": 350.0
      }
    },
    {
      "class": "coco_dog",
      "class_id": 16,
      "confidence": 0.87,
      "bounding_box": {
        "x1": 400.0,
        "y1": 200.0,
        "x2": 600.0,
        "y2": 500.0,
        "width": 200.0,
        "height": 300.0
      }
    }
  ],
  "class_probabilities": {
    "coco_person": 0.95,
    "coco_dog": 0.87
  },
  "detection_count": 2
}
```

