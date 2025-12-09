# YOLO Training and Inference Guide

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download your Roboflow dataset:**
   - Go to your Roboflow project
   - Click "Export Dataset"
   - Select "YOLO v8" format
   - Download and extract to your project directory

## Training

### Basic Training Command

```bash
python train_yolo.py
```

**Before running, edit `train_yolo.py` and update:**
- `DATA_YAML`: Path to your dataset's `data.yaml` file (e.g., `"dataset/data.yaml"`)
- `MODEL_NAME`: Choose model size (`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, etc.)
- `EPOCHS`: Number of training epochs (default: 100)
- `BATCH_SIZE`: Batch size (adjust based on your GPU memory)

### Model Sizes

- `yolov8n.pt` - Nano (fastest, least accurate)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium (good balance)
- `yolov8l.pt` - Large
- `yolov8x.pt` - XLarge (slowest, most accurate)

### Training Output

Results will be saved to:
- `runs/detect/yolo_custom_train/weights/best.pt` - Best model checkpoint
- `runs/detect/yolo_custom_train/weights/last.pt` - Last epoch checkpoint
- Training metrics, confusion matrix, and plots in the same directory

## Inference

### Option 1: Using inference.py

```bash
# Basic usage
python inference.py path/to/image.jpg

# With custom model and confidence threshold
python inference.py path/to/image.jpg runs/detect/yolo_custom_train/weights/best.pt 0.5
```

### Option 2: Quick Command Line

```bash
# Using ultralytics CLI
yolo predict model=runs/detect/yolo_custom_train/weights/best.pt source=path/to/image.jpg conf=0.25
```

### Option 3: In Python Script

```python
from ultralytics import YOLO

# Load model
model = YOLO('runs/detect/yolo_custom_train/weights/best.pt')

# Run inference
results = model.predict('image.jpg', conf=0.25, save=True)

# Get detections
for result in results:
    boxes = result.boxes
    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]
        print(f"{class_name}: {confidence:.2f}")
```

## Inference on Different Sources

### Single Image
```python
results = model.predict('image.jpg')
```

### Multiple Images
```python
results = model.predict(['image1.jpg', 'image2.jpg', 'image3.jpg'])
```

### Folder of Images
```python
results = model.predict('path/to/folder/')
```

### Video
```python
results = model.predict('video.mp4', save=True)
```

### Webcam (Real-time)
```python
results = model.predict(source=0, show=True, stream=True)
```

### YouTube Video
```python
results = model.predict('https://www.youtube.com/watch?v=VIDEO_ID')
```

### RTSP Stream
```python
results = model.predict('rtsp://username:password@ip_address:port/stream')
```

## Advanced Training Options

### Train with Custom Parameters

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.01,  # Initial learning rate
    optimizer='Adam',  # Optimizer (SGD, Adam, AdamW)
    augment=True,  # Use augmentation
    patience=50,  # Early stopping patience
    save_period=10,  # Save checkpoint every N epochs
    device=0,  # GPU device (0, 1, 2, etc.) or 'cpu'
    workers=8,  # Number of worker threads
    project='runs/detect',
    name='custom_experiment',
    exist_ok=True
)
```

### Resume Training

```python
model = YOLO('runs/detect/yolo_custom_train/weights/last.pt')
model.train(resume=True)
```

### Fine-tune Pre-trained Model

```python
# Start from a model trained on your dataset
model = YOLO('runs/detect/yolo_custom_train/weights/best.pt')

# Continue training with more epochs
model.train(data='dataset/data.yaml', epochs=50)
```

## Model Validation

```python
from ultralytics import YOLO

model = YOLO('runs/detect/yolo_custom_train/weights/best.pt')

# Validate on test set
metrics = model.val(data='dataset/data.yaml')

print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")
print(f"mAP75: {metrics.box.map75}")
```

## Export Model

### Export to ONNX (for deployment)
```python
model = YOLO('runs/detect/yolo_custom_train/weights/best.pt')
model.export(format='onnx')
```

### Other Export Formats
```python
# TensorRT (for NVIDIA GPUs)
model.export(format='engine')

# CoreML (for iOS)
model.export(format='coreml')

# TensorFlow
model.export(format='pb')
model.export(format='tflite')
```

## Troubleshooting

### Out of Memory Error
- Reduce batch size: `batch=8` or `batch=4`
- Use smaller model: `yolov8n.pt` instead of `yolov8m.pt`
- Reduce image size: `imgsz=416` instead of `imgsz=640`

### Poor Performance
- Train for more epochs
- Use a larger model
- Check your dataset quality and annotations
- Increase image size: `imgsz=1280`
- Adjust confidence threshold during inference

### CPU Training (No GPU)
Edit `train_yolo.py` and change:
```python
device='cpu'
```

## Monitoring Training

View training metrics in real-time:
- TensorBoard: Training logs are automatically saved
- Check `runs/detect/yolo_custom_train/` for plots and metrics

```bash
# View in TensorBoard (optional)
tensorboard --logdir runs/detect
```

## Tips for Best Results

1. **Dataset Quality**: Ensure good annotations and diverse images
2. **Training Time**: More epochs generally = better results (with early stopping)
3. **Model Size**: Balance between speed and accuracy
4. **Augmentation**: Keep enabled for better generalization
5. **Validation**: Always validate on a separate test set
6. **Confidence Threshold**: Adjust during inference (0.25 default, higher = fewer false positives)
