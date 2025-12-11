# Card Detection Guide

This guide shows you how to use the trained YOLOv8 model to detect playing cards in images and videos.

## Prerequisites

- Trained model at: `runs/detect/yolo_custom_train/weights/best.pt`
- Dependencies installed: `pip install -r requirements.txt`

---

## Detecting Cards in Images

### Single Image Detection

Detect cards in one image and save the annotated result:

```powershell
py detect_image.py path/to/your/image.jpg
```

**Example:**
```powershell
py detect_image.py test_cards.jpg
```

**Output:**
- Annotated image saved to: `examples/result/images/test_cards_annotated.jpg`
- Console displays detected cards with confidence scores

**Sample Output:**
```
DETECTION RESULTS
============================================================
Image: test_cards.jpg
Cards detected: 3
============================================================

Detection 1:
  Card: 7H
  Confidence: 0.987
  Bounding Box: [120.5, 45.3, 280.1, 350.8]

Detection 2:
  Card: KS
  Confidence: 0.956
  Bounding Box: [350.2, 60.7, 510.4, 365.2]

Detection 3:
  Card: AC
  Confidence: 0.992
  Bounding Box: [540.1, 55.2, 700.3, 360.5]

============================================================
Annotated image saved to: examples/result/images/test_cards_annotated.jpg
============================================================
```

### Custom Confidence Threshold

Adjust the confidence threshold (default is 0.25):

```powershell
# Higher confidence = fewer detections (more strict)
py detect_image.py image.jpg 0.5

# Lower confidence = more detections (less strict)
py detect_image.py image.jpg 0.15
```

**When to adjust:**
- **0.5-0.7** - High precision, may miss some cards
- **0.25-0.4** - Balanced (recommended)
- **0.1-0.2** - High recall, may have false positives

### Custom Output Directory

Save results to a specific folder (default is `examples/result/images/`):

```powershell
py detect_image.py image.jpg 0.25 my_results/
```

### Batch Process Multiple Images

Process all images in a folder at once:

```powershell
# Process all images in a directory
py detect_image.py path/to/image/folder/ batch
```

**Example:**
```powershell
py detect_image.py test_images/ batch
```

**Output:**
- Processes all .jpg, .png, .bmp files in the folder
- Saves all annotated images to `examples/result/images/` folder
- Shows summary statistics

---

## Detecting Cards in Videos

### Single Video Processing

Detect cards in a video file:

```powershell
py detect_video.py path/to/video.mp4
```

**Example:**
```powershell
py detect_video.py card_game.mp4
```

**Output:**
- Annotated video saved to: `examples/result/video/card_game.mp4`
- Console shows frame-by-frame detection counts
- Summary statistics at the end

**Sample Output:**
```
Processing video: card_game.mp4
Confidence threshold: 0.25

Frame 1: 5 cards detected
  - 2H: 0.95
  - 7D: 0.89
  - KS: 0.97
  - AC: 0.93
  - 10C: 0.91

Frame 2: 5 cards detected
...

Processing complete!
Total frames: 120
Total detections: 600
Average detections per frame: 5.00
Output video saved to: examples/result/video/
```

### Custom Confidence for Video

```powershell
py detect_video.py video.mp4 0.4
```

### Real-Time Webcam Detection

Detect cards from your webcam in real-time:

```powershell
py detect_video.py webcam
```

**Controls:**
- Press **'q'** to quit
- Detections shown in real-time with bounding boxes and labels

**Use cases:**
- Live card game analysis
- Testing model performance
- Demo presentations

### Batch Process Multiple Videos

Process all videos in a folder:

```powershell
py detect_video.py batch path/to/video/folder/
```

**Supported formats:** .mp4, .avi, .mov, .mkv, .flv

---

## Alternative Methods

### Using YOLO CLI Directly

#### Images
```powershell
yolo predict model=runs/detect/yolo_custom_train/weights/best.pt source=image.jpg save=True
```

#### Videos
```powershell
yolo predict model=runs/detect/yolo_custom_train/weights/best.pt source=video.mp4 save=True
```

#### Webcam
```powershell
yolo predict model=runs/detect/yolo_custom_train/weights/best.pt source=0 show=True
```

### Using Python Code

For custom integration into your own projects:

```python
from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO('runs/detect/yolo_custom_train/weights/best.pt')

# Detect in an image
results = model.predict('image.jpg', conf=0.25, save=True)

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        # Get card information
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        card_name = result.names[class_id]
        
        # Get bounding box coordinates [x1, y1, x2, y2]
        bbox = box.xyxy[0].tolist()
        
        print(f"Detected: {card_name} ({confidence:.2f})")
        print(f"Location: {bbox}")
```

---

## Understanding the Output

### Card Labels

Cards are labeled with rank and suit:
- **Format:** `<Rank><Suit>`
- **Ranks:** 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A
- **Suits:** C (Clubs), D (Diamonds), H (Hearts), S (Spades)

**Examples:**
- `7H` = 7 of Hearts
- `KS` = King of Spades
- `AC` = Ace of Clubs
- `10D` = 10 of Diamonds

### Confidence Score

The confidence score (0.0 to 1.0) indicates how certain the model is:
- **0.9-1.0** - Very confident (excellent detection)
- **0.7-0.9** - Confident (good detection)
- **0.5-0.7** - Moderate confidence
- **0.25-0.5** - Low confidence (may be incorrect)

### Bounding Box

The bounding box `[x1, y1, x2, y2]` defines the rectangle around the card:
- `x1, y1` - Top-left corner coordinates
- `x2, y2` - Bottom-right corner coordinates
- Coordinates are in pixels from the top-left of the image

---

## Tips and Best Practices

### For Best Detection Results:

1. **Good Lighting:** Ensure cards are well-lit and visible
2. **Clear Images:** Avoid blurry or low-resolution images
3. **Card Visibility:** Cards should be fully visible (not heavily occluded)
4. **Appropriate Angle:** Front-facing cards work best

### Performance Optimization:

**For faster processing:**
- Use lower resolution images/videos when possible
- Increase confidence threshold to reduce processing time
- Use GPU if available (automatically detected)

**For better accuracy:**
- Use higher confidence threshold (0.4-0.5)
- Ensure good image quality
- Process images at native resolution

### Troubleshooting:

**No detections found:**
- Lower the confidence threshold: `py detect_image.py image.jpg 0.15`
- Check image quality and lighting
- Ensure cards are clearly visible

**Too many false detections:**
- Raise the confidence threshold: `py detect_image.py image.jpg 0.5`
- Check if model needs retraining with more diverse data

**Slow processing:**
- Check if GPU is being used (should see "CUDA:0" in output)
- Reduce video resolution
- Process fewer images at once

---

## Quick Reference

| Task | Command |
|------|---------|
| Single image | `py detect_image.py image.jpg` |
| Image with custom confidence | `py detect_image.py image.jpg 0.5` |
| Batch images | `py detect_image.py folder/ batch` |
| Single video | `py detect_video.py video.mp4` |
| Webcam | `py detect_video.py webcam` |
| Batch videos | `py detect_video.py batch folder/` |

---

## Model Information

**Your Trained Model:**
- **Location:** `runs/detect/yolo_custom_train/weights/best.pt`
- **Classes:** 52 playing cards (full deck)
- **Performance:** 
  - mAP50-95: 83.6%
  - Precision: 99.9%
  - Recall: 100%
- **Architecture:** YOLOv8n (Nano)

This model can detect all 52 standard playing cards with high accuracy and speed!
