"""
Single Image Card Detection
Runs the trained model on a single image and saves annotated result
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import sys

def detect_and_annotate(image_path, model_path='runs/detect/yolo_custom_train/weights/best.pt', 
                       output_dir='examples/result/images', conf_threshold=0.25):
    """
    Run card detection on a single image and save annotated result.
    
    Args:
        image_path: Path to input image
        model_path: Path to trained YOLO model
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Load image
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    print(f"Processing image: {image_path.name}")
    print(f"Confidence threshold: {conf_threshold}")
    
    # Run detection
    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        save=False,
        verbose=False
    )
    
    # Get result
    result = results[0]
    boxes = result.boxes
    
    print(f"\n{'='*60}")
    print(f"DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"Image: {image_path.name}")
    print(f"Cards detected: {len(boxes)}")
    print(f"{'='*60}\n")
    
    # Print detections
    if len(boxes) > 0:
        for i, box in enumerate(boxes, 1):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            
            print(f"Detection {i}:")
            print(f"  Card: {class_name}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Bounding Box: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
            print()
    else:
        print("No cards detected in this image.\n")
    
    # Get annotated image
    annotated_img = result.plot()
    
    # Save annotated image
    output_file = output_path / f"{image_path.stem}_annotated{image_path.suffix}"
    cv2.imwrite(str(output_file), annotated_img)
    
    print(f"{'='*60}")
    print(f"Annotated image saved to: {output_file}")
    print(f"{'='*60}")
    
    return str(output_file), len(boxes)


def batch_detect_images(image_folder, model_path='runs/detect/yolo_custom_train/weights/best.pt',
                        output_dir='examples/result/images', conf_threshold=0.25):
    """
    Process all images in a folder.
    
    Args:
        image_folder: Path to folder containing images
        model_path: Path to trained YOLO model
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
    """
    folder = Path(image_folder)
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        return
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(folder.glob(f'*{ext}'))
        image_files.extend(folder.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No images found in: {folder}")
        return
    
    print(f"\nFound {len(image_files)} images to process\n")
    
    # Process each image
    total_detections = 0
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] " + "="*50)
        output_file, num_detections = detect_and_annotate(
            image_path, model_path, output_dir, conf_threshold
        )
        total_detections += num_detections
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Images processed: {len(image_files)}")
    print(f"Total cards detected: {total_detections}")
    print(f"Average cards per image: {total_detections/len(image_files):.1f}")
    print(f"Results saved to: {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Default settings
    MODEL_PATH = "runs/detect/yolo_custom_train/weights/best.pt"
    OUTPUT_DIR = "examples/result/images"
    CONFIDENCE = 0.25
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image:  py detect_image.py path/to/image.jpg")
        print("  Batch folder:  py detect_image.py path/to/folder/ batch")
        print("  Custom conf:   py detect_image.py image.jpg 0.5")
        print("  Custom output: py detect_image.py image.jpg 0.25 output_folder/")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Check if batch mode
    is_batch = len(sys.argv) > 2 and sys.argv[2].lower() == 'batch'
    
    # Get optional confidence threshold
    if len(sys.argv) > 2 and sys.argv[2].replace('.', '').isdigit():
        CONFIDENCE = float(sys.argv[2])
    
    # Get optional output directory
    if len(sys.argv) > 3:
        OUTPUT_DIR = sys.argv[3]
    
    # Process
    if is_batch or Path(input_path).is_dir():
        batch_detect_images(input_path, MODEL_PATH, OUTPUT_DIR, CONFIDENCE)
    else:
        detect_and_annotate(input_path, MODEL_PATH, OUTPUT_DIR, CONFIDENCE)
