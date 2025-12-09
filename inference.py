"""
Simple inference script for trained YOLO model
"""

from ultralytics import YOLO
import cv2
import sys

def predict_image(model_path, image_path, conf_threshold=0.25):
    """
    Run YOLO inference on a single image and display results.
    
    Args:
        model_path: Path to trained model weights
        image_path: Path to input image
        conf_threshold: Confidence threshold for detections
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Run inference
    print(f"Running inference on {image_path}...")
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=True,
        show=False
    )
    
    # Process results
    for result in results:
        # Get detection info
        boxes = result.boxes
        print(f"\n{'='*50}")
        print(f"Image: {image_path}")
        print(f"Detections: {len(boxes)}")
        print(f"{'='*50}")
        
        # Print each detection
        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            
            print(f"\nDetection {i+1}:")
            print(f"  Class: {class_name}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Bounding Box: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        
        # Save annotated image
        annotated = result.plot()
        output_path = f"prediction_{image_path.split('/')[-1]}"
        cv2.imwrite(output_path, annotated)
        print(f"\nAnnotated image saved to: {output_path}")
    
    return results


def predict_video(model_path, video_path, conf_threshold=0.25):
    """
    Run YOLO inference on a video.
    
    Args:
        model_path: Path to trained model weights
        video_path: Path to input video
        conf_threshold: Confidence threshold for detections
    """
    model = YOLO(model_path)
    
    results = model.predict(
        source=video_path,
        conf=conf_threshold,
        save=True,
        stream=True  # Use streaming for videos
    )
    
    for result in results:
        boxes = result.boxes
        print(f"Frame detections: {len(boxes)}")


def predict_webcam(model_path, conf_threshold=0.25):
    """
    Run YOLO inference on webcam feed.
    
    Args:
        model_path: Path to trained model weights
        conf_threshold: Confidence threshold for detections
    """
    model = YOLO(model_path)
    
    # Run inference on webcam (source=0 for default webcam)
    results = model.predict(
        source=0,
        conf=conf_threshold,
        show=True,
        stream=True
    )
    
    for result in results:
        pass  # Results are displayed in real-time


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "runs/detect/yolo_custom_train/weights/best.pt"
    IMAGE_PATH = "path/to/your/test/image.jpg"
    CONFIDENCE = 0.25
    
    # Check command line arguments
    if len(sys.argv) > 1:
        IMAGE_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        MODEL_PATH = sys.argv[2]
    if len(sys.argv) > 3:
        CONFIDENCE = float(sys.argv[3])
    
    # Run inference
    print("="*60)
    print("YOLO Inference Script")
    print("="*60)
    
    predict_image(MODEL_PATH, IMAGE_PATH, CONFIDENCE)
    
    # Uncomment to use video or webcam:
    # predict_video(MODEL_PATH, "path/to/video.mp4", CONFIDENCE)
    # predict_webcam(MODEL_PATH, CONFIDENCE)
