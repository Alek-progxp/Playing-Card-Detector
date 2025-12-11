# Playing Card Detector

A YOLOv8-based object detection system that identifies and classifies all 52 playing cards by rank and suit in images and videos.

## Features

- ✅ Detects all 52 playing cards (ranks: 2-10, J, Q, K, A | suits: ♠ ♥ ♦ ♣)
- ✅ High accuracy: 99.9% precision, 100% recall, 83.6% mAP50-95
- ✅ Real-time detection in videos and webcam feeds
- ✅ Batch processing for multiple images/videos
- ✅ GPU-accelerated inference

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Using the Pre-trained Model

The trained model is located at: `runs/detect/yolo_custom_train/weights/best.pt`

#### Detect Cards in Images

```bash
# Single image
py detect_image.py path/to/image.jpg

# Batch process folder
py detect_image.py path/to/folder/ batch
```

**Output:** Annotated images saved to `examples/result/images/`

#### Detect Cards in Videos

```bash
# Process video
py detect_video.py path/to/video.mp4

# Real-time webcam (press 'q' to quit)
py detect_video.py webcam
```

**Output:** Annotated videos saved to `examples/result/video/`

### 3. Adjusting Confidence Threshold

```bash
# Higher confidence = more strict (fewer false positives)
py detect_image.py image.jpg 0.5

# Lower confidence = more lenient (catches more cards)
py detect_video.py video.mp4 0.2
```

**Recommended:** 0.25 (default) for balanced performance

## Card Labels

Cards are labeled as: `<Rank><Suit>`

- **Ranks:** 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A
- **Suits:** C (♣ Clubs), D (♦ Diamonds), H (♥ Hearts), S (♠ Spades)

**Examples:** `7H` = 7 of Hearts, `KS` = King of Spades, `AC` = Ace of Clubs

## Model Performance

| Metric | Value |
|--------|-------|
| mAP50-95 | 83.6% |
| mAP50 | 99.5% |
| Precision | 99.9% |
| Recall | 100% |

Trained on 21,203 images with 52 classes using YOLOv8n.

## Project Structure

```
Playing-Card-Detector/
├── detect_image.py          # Image detection script
├── detect_video.py          # Video detection script
├── train_yolo.py            # Model training script
├── analyze_results.py       # Training results analysis
├── runs/detect/             # Training outputs and models
│   └── yolo_custom_train/
│       └── weights/
│           └── best.pt      # Trained model
├── DETECTION_GUIDE.md       # Detailed usage guide
└── USAGE.md                 # Full training documentation
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM for training, 4GB+ for inference

## Documentation

- **[DETECTION_GUIDE.md](DETECTION_GUIDE.md)** - Comprehensive detection usage guide
- **[USAGE.md](USAGE.md)** - Complete training and inference documentation

## License

 - This project uses the [Playing Cards Dataset](https://universe.roboflow.com/augmented-startups/playing-cards-ow27d) from Roboflow (Public Domain).
 - This project uses a [pretrained YOLOv8 model](https://docs.ultralytics.com/models/yolov8/) from Ultralytics (Free to use under AGPL-3.0 License).