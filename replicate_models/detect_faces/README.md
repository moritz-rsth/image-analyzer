# Face Detection Model (MTCNN)

Dieses Modell erkennt Gesichter in Bildern mit dem MTCNN (Multi-task Cascaded Convolutional Networks) Modell.

## Hardware-Empfehlung
- **Empfohlen:** GPU (Nvidia T4 oder besser)
- **Minimum:** CPU (funktioniert, aber langsam)

## Installation und Deployment

### 1. Model auf Replicate erstellen
Besuche https://replicate.com/create und erstelle ein neues Modell mit dem Namen `your-username/detect-faces`

### 2. Mit Cog builden und pushen
```bash
cd replicate_models/detect_faces
cog build -t your-username/detect-faces
cog push your-username/detect-faces
```

### 3. Deployment erstellen
Nach dem Push kannst du auf Replicate ein Deployment erstellen und die Hardware auswählen.

## Inputs
- `image` (Path): Bilddatei zur Gesichtserkennung
- `confidence_threshold` (float, optional): Confidence-Schwelle für Gesichtserkennung (0.0-1.0, Standard: 0.85)

## Output
JSON-String mit folgenden Feldern:
- `face_count`: Anzahl der erkannten Gesichter
- `face_confidence_avg`: Durchschnittliche Confidence
- `face_area_total_abs`: Gesamte absolute Gesichtsfläche in Pixeln
- `face_area_total_rel`: Gesamte relative Gesichtsfläche (0-1)
- `faces`: Liste der erkannten Gesichter mit Bounding Boxes und Landmarks

## Beispiel-Output
```json
{
  "face_count": 2,
  "face_confidence_avg": 0.95,
  "face_area_total_abs": 50000,
  "face_area_total_rel": 0.15,
  "faces": [
    {
      "face_id": 1,
      "confidence": 0.96,
      "bounding_box": {"x": 100, "y": 150, "width": 200, "height": 250},
      "landmarks": {
        "right_eye": [150, 200],
        "left_eye": [250, 200],
        "nose": [200, 250],
        "mouth_right": [220, 300],
        "mouth_left": [180, 300]
      }
    }
  ]
}
```

