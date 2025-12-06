# ImageNet Classification Model (YOLO11)

Dieses Modell klassifiziert Bilder mit dem YOLO11 Classification Modell und gibt ImageNet-Klassen zurück.

## Hardware-Empfehlung
- **Empfohlen:** GPU (Nvidia T4 oder besser)
- **Minimum:** GPU erforderlich (CPU ist zu langsam)

## Installation und Deployment

### 1. Model auf Replicate erstellen
Besuche https://replicate.com/create und erstelle ein neues Modell mit dem Namen `your-username/predict-imagenet-classes-yolo11`

### 2. Mit Cog builden und pushen
```bash
cd replicate_models/predict_imagenet_classes_yolo11
cog build -t your-username/predict-imagenet-classes-yolo11
cog push your-username/predict-imagenet-classes-yolo11
```

### 3. Deployment erstellen
Nach dem Push kannst du auf Replicate ein Deployment erstellen und die Hardware auswählen.

## Inputs
- `image` (Path): Bilddatei zur Klassifikation
- `top_k` (int, optional): Anzahl der Top-Prädiktionen zurückgeben (1-1000, Standard: 10)

## Output
JSON-String mit folgenden Feldern:
- `predictions`: Liste der Top-K Prädiktionen mit Klassenname, Wahrscheinlichkeit und Class-ID
- `all_classes`: Dictionary mit allen ImageNet-Klassen und ihren Wahrscheinlichkeiten

## Beispiel-Output
```json
{
  "predictions": [
    {
      "class": "golden_retriever",
      "probability": 0.85,
      "class_id": 207
    },
    {
      "class": "labrador_retriever",
      "probability": 0.12,
      "class_id": 208
    }
  ],
  "all_classes": {
    "imagenet_golden_retriever": 0.85,
    "imagenet_labrador_retriever": 0.12,
    ...
  }
}
```

