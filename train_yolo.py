"""
YOLO Training Script for Custom Roboflow Dataset
This script trains a YOLOv8 model on a custom dataset and runs inference.
"""

from ultralytics import YOLO

def train_model(data_yaml_path, model_name='yolov8n.pt', epochs=100, imgsz=640, batch=16):
    """
    Train YOLO model on custom dataset.
    
    Args:
        data_yaml_path: Path to data.yaml file from Roboflow
        model_name: Pretrained model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
    """
    # Load a pretrained YOLO model
    model = YOLO(model_name)
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=50,  # Early stopping patience
        save=True,
        device=0,  # Use GPU 0, change to 'cpu' if no GPU available
        workers=8,
        project='runs/detect',
        name='yolo_custom_train',
        exist_ok=True
    )
    
    return model, results


def validate_model(model, data_yaml_path):
    """
    Validate the trained model.
    
    Args:
        model: Trained YOLO model
        data_yaml_path: Path to data.yaml file
    """
    metrics = model.val(data=data_yaml_path)
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP75: {metrics.box.map75}")
    return metrics


def run_inference(model_path, image_path, conf_threshold=0.25, save=True):
    """
    Run inference on a new image.
    
    Args:
        model_path: Path to trained model weights (best.pt or last.pt)
        image_path: Path to image for inference
        conf_threshold: Confidence threshold for detections
        save: Whether to save results
    """
    # Load the trained model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=save,
        project='runs/detect',
        name='predict',
        exist_ok=True
    )
    
    # Print results
    for result in results:
        boxes = result.boxes
        print(f"\nDetections for {image_path}:")
        print(f"Number of objects detected: {len(boxes)}")
        
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]
            print(f"  - {class_name}: {confidence:.2f}")
    
    return results


def export_model(model_path, format='onnx'):
    """
    Export model to different formats.
    
    Args:
        model_path: Path to trained model weights
        format: Export format (onnx, torchscript, coreml, etc.)
    """
    model = YOLO(model_path)
    model.export(format=format)
    print(f"Model exported to {format} format")


if __name__ == "__main__":
    # ========== CONFIGURATION ==========
    # Path to your Roboflow dataset's data.yaml file
    # After downloading from Roboflow, this will be in your dataset folder
    DATA_YAML = r"Playing Cards.v2i.yolov8\data.yaml"
    
    # Choose model size: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)
    # yolov8l.pt (large), yolov8x.pt (xlarge)
    MODEL_NAME = "yolov8n.pt"
    
    # Training parameters
    EPOCHS = 100
    IMAGE_SIZE = 640
    BATCH_SIZE = 16
    
    # ========== TRAINING ==========
    print("Starting YOLO training...")
    print(f"Dataset: {DATA_YAML}")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}, Image size: {IMAGE_SIZE}, Batch: {BATCH_SIZE}")
    
    # Train the model
    model, results = train_model(
        data_yaml_path=DATA_YAML,
        model_name=MODEL_NAME,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE
    )
    
    # ========== VALIDATION ==========
    print("\nValidating model...")
    metrics = validate_model(model, DATA_YAML)
    
    # ========== INFERENCE ==========
    # Path to your trained model (usually in runs/detect/yolo_custom_train/weights/best.pt)
    TRAINED_MODEL = "runs/detect/yolo_custom_train/weights/best.pt"
    
    # Path to test image
    TEST_IMAGE = "path/to/your/test/image.jpg"
    
    # Run inference (uncomment when you have a test image)
    # print(f"\nRunning inference on {TEST_IMAGE}...")
    # results = run_inference(TRAINED_MODEL, TEST_IMAGE, conf_threshold=0.25)
    
    # ========== EXPORT (Optional) ==========
    # Export model to ONNX format for deployment
    # export_model(TRAINED_MODEL, format='onnx')
    
    print("\nTraining complete!")
    print(f"Best model saved to: {TRAINED_MODEL}")
