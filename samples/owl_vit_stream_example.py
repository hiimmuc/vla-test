#!/usr/bin/env python3
"""
OWL-ViT Object Detection for Images, Videos, and Webcam
Supports user-defined object detection with natural language prompts.
"""

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "google/owlvit-base-patch32"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.2
PROCESSING_FRAME_SKIP = 1  # Process every 3rd frame for better performance
DETECTION_CACHE_TIME = 0.1  # Cache detections for 0.5 seconds


class FPSCounter:
    """Simple FPS counter for performance monitoring."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()

    def update(self) -> float:
        """Update FPS counter and return current FPS."""
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time

        # Keep only recent frame times
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)

        # Calculate FPS
        if len(self.frame_times) > 1:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        return 0.0


class OWLViTObjectDetector:
    """OWL-ViT based object detector with real-time capabilities."""

    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME, force_download=False)
        self.processor.save_pretrained("checkpoints/" + MODEL_NAME)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            MODEL_NAME, force_download=False
        )
        self.model.save_pretrained("checkpoints/" + MODEL_NAME)

        self.model.to(DEVICE)
        self.model.eval()

        # Thread-safe variables for real-time object updates
        self._target_objects = []
        self._objects_lock = threading.Lock()

        # Performance optimization: Cache results
        self._detection_cache = None
        self._cache_timestamp = 0
        self._cache_lock = threading.Lock()

        logger.info(f"Initialized OWL-ViT model: {MODEL_NAME}")

    def update_target_objects(self, new_objects: List[str]) -> None:
        """Thread-safe method to update target objects during runtime."""
        with self._objects_lock:
            self._target_objects = [obj.strip() for obj in new_objects if obj.strip()]
            logger.info(f"Updated target objects: {self._target_objects}")

    def get_target_objects(self) -> List[str]:
        """Thread-safe method to get current target objects."""
        with self._objects_lock:
            return self._target_objects.copy()

    def detect_objects(
        self, image: np.ndarray, target_objects: List[str], use_cache: bool = True
    ) -> List[Tuple[np.ndarray, str, float]]:
        """Detect specified objects in the image using OWL-ViT."""
        if not target_objects:
            return []

        current_time = time.time()

        # Check cache first for streaming performance
        if use_cache:
            with self._cache_lock:
                if (
                    self._detection_cache is not None
                    and current_time - self._cache_timestamp < DETECTION_CACHE_TIME
                ):
                    return self._detection_cache

        # Convert BGR to RGB
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Resize for better performance
        original_size = pil_image.size
        if max(original_size) > 768:
            pil_image.thumbnail((768, 768), Image.Resampling.LANCZOS)

        # Prepare text prompts
        text_queries = [[f"a photo of {obj}" for obj in target_objects]]

        # Process inputs
        inputs = self.processor(text=text_queries, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        target_sizes = torch.Tensor([pil_image.size[::-1]]).to(DEVICE)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=CONFIDENCE_THRESHOLD, target_sizes=target_sizes
        )

        # Extract detections
        detections = []
        if results:
            boxes = results[0]["boxes"]
            scores = results[0]["scores"]
            labels = results[0]["labels"]

            # Scale boxes back to original image size if resized
            scale_x = original_size[0] / pil_image.size[0]
            scale_y = original_size[1] / pil_image.size[1]

            for box, score, label_idx in zip(boxes, scores, labels):
                if label_idx < len(target_objects):
                    # Scale box coordinates
                    box_scaled = box.cpu().numpy()
                    box_scaled[0] *= scale_x  # x1
                    box_scaled[1] *= scale_y  # y1
                    box_scaled[2] *= scale_x  # x2
                    box_scaled[3] *= scale_y  # y2

                    detections.append((box_scaled, target_objects[label_idx], score.item()))

        # Update cache
        if use_cache:
            with self._cache_lock:
                self._detection_cache = detections
                self._cache_timestamp = current_time

        return detections


def draw_detections(
    image: np.ndarray, detections: List[Tuple[np.ndarray, str, float]], fps: float = 0.0
) -> np.ndarray:
    """Draw bounding boxes, labels, and FPS on the image."""
    result_image = image.copy()

    # Draw FPS in top-right corner
    if fps > 0:
        fps_text = f"FPS: {fps:.1f}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(
            result_image,
            (result_image.shape[1] - text_size[0] - 20, 10),
            (result_image.shape[1] - 10, text_size[1] + 30),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            result_image,
            fps_text,
            (result_image.shape[1] - text_size[0] - 15, text_size[1] + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    for box, label, confidence in detections:
        x1, y1, x2, y2 = map(int, box)

        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label with confidence
        label_text = f"{label}: {confidence:.2f}"
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        # Background rectangle for text
        cv2.rectangle(
            result_image,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            (0, 255, 0),
            -1,
        )

        # Text
        cv2.putText(
            result_image,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    return result_image


def process_image(
    detector: OWLViTObjectDetector, image_path: str, target_objects: List[str]
) -> None:
    """Process a single image."""
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        return

    logger.info(f"Processing image: {image_path}")
    detections = detector.detect_objects(image, target_objects, use_cache=False)

    if detections:
        logger.info(f"Found {len(detections)} objects")
        result_image = draw_detections(image, detections)

        # Save result
        output_path = Path(image_path).with_suffix(".detected.jpg")
        cv2.imwrite(str(output_path), result_image)
        logger.info(f"Saved result to: {output_path}")

        # Display result
        cv2.imshow("Detection Result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        logger.info("No objects detected")


def process_video(
    detector: OWLViTObjectDetector, video_path: str, target_objects: List[str]
) -> None:
    """Process a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return

    logger.info(f"Processing video: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup video writer
    output_path = Path(video_path).with_suffix(".detected.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    fps_counter = FPSCounter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_fps = fps_counter.update()

        if frame_count % 30 == 0:  # Process every 30th frame for efficiency
            logger.info(f"Processing frame {frame_count} - FPS: {current_fps:.1f}")

        detections = detector.detect_objects(frame, target_objects, use_cache=False)
        result_frame = draw_detections(frame, detections, current_fps)

        out.write(result_frame)

        # Display frame
        cv2.imshow("Video Detection", result_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logger.info(f"Saved processed video to: {output_path}")


def process_webcam(detector: OWLViTObjectDetector, target_objects: List[str]) -> None:
    """Process webcam stream with real-time object input and performance monitoring."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam")
        return

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency

    # Initialize target objects
    detector.update_target_objects(target_objects)

    # Start input thread for real-time object updates
    input_thread = threading.Thread(target=input_handler, args=(detector,), daemon=True)
    input_thread.start()

    # Initialize FPS counter
    fps_counter = FPSCounter()

    logger.info("Starting webcam detection with OWL-ViT.")
    logger.info("Commands:")
    logger.info("  Type object names to detect (e.g., 'cat dog car')")
    logger.info("  Type 'clear' to clear all objects")
    logger.info("  Type 'list' to show current objects")
    logger.info("  Press 'q' in the video window to quit")
    print("Enter objects to detect: ", end="", flush=True)

    frame_skip = 0
    last_detections = []
    last_detection_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read from webcam")
            break

        # Update FPS counter
        fps = fps_counter.update()

        current_objects = detector.get_target_objects()
        current_time = time.time()

        # Process detection only if needed (performance optimization)
        should_detect = (
            frame_skip % PROCESSING_FRAME_SKIP == 0
            and current_objects
            and (current_time - last_detection_time > DETECTION_CACHE_TIME / 2)
        )

        if should_detect:
            detections = detector.detect_objects(frame, current_objects, use_cache=True)
            last_detections = detections
            last_detection_time = current_time

        # Draw detections and FPS on every frame for smooth display
        frame = draw_detections(frame, last_detections, fps)

        # Add instructions overlay
        add_instructions_overlay(frame, current_objects, fps)

        frame_skip += 1

        cv2.imshow("OWL-ViT Webcam Detection - Real-time Object Input", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def add_instructions_overlay(
    frame: np.ndarray, current_objects: List[str], fps: float = 0.0
) -> None:
    """Add instructions and current objects overlay to the frame."""
    height, width = frame.shape[:2]

    # Semi-transparent overlay
    overlay = frame.copy()

    # Instructions box (adjusted height for FPS display)
    cv2.rectangle(overlay, (10, 10), (width - 150, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Instructions text
    instructions = [
        "Commands: Type objects to detect | 'clear' | 'list' | 'q' to quit",
        f"Currently detecting: {', '.join(current_objects) if current_objects else 'None'}",
        f"Objects found: {len(current_objects)} types",
        f"Performance: {fps:.1f} FPS" if fps > 0 else "Performance: Calculating...",
    ]

    for i, text in enumerate(instructions):
        cv2.putText(
            frame,
            text,
            (15, 30 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def input_handler(detector: OWLViTObjectDetector) -> None:
    """Handle user input in a separate thread for real-time object updates."""
    while True:
        try:
            user_input = input().strip().lower()

            if user_input == "quit" or user_input == "q":
                break
            elif user_input == "clear":
                detector.update_target_objects([])
                print("Cleared all objects.")
            elif user_input == "list":
                current = detector.get_target_objects()
                print(f"Current objects: {', '.join(current) if current else 'None'}")
            elif user_input:
                # Parse new objects (space or comma separated)
                new_objects = user_input.replace(",", " ").split()
                detector.update_target_objects(new_objects)

            print("Enter objects to detect: ", end="", flush=True)

        except EOFError:
            break
        except KeyboardInterrupt:
            break


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OWL-ViT Object Detection for Images, Videos, and Webcam"
    )
    parser.add_argument("--input", "-i", type=str, help="Input image or video path")
    parser.add_argument(
        "--webcam", "-w", action="store_true", help="Use webcam input with real-time object input"
    )
    parser.add_argument(
        "--objects",
        "-o",
        type=str,
        help="Initial comma-separated list of objects to detect (e.g., 'cat,dog,car'). For webcam mode, you can also input objects during runtime.",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Parse initial target objects
    initial_objects = []
    if args.objects:
        initial_objects = [obj.strip() for obj in args.objects.split(",")]
        logger.info(f"Initial target objects: {initial_objects}")

    # Initialize detector
    detector = OWLViTObjectDetector()

    if args.webcam:
        # For webcam mode, allow empty initial objects
        process_webcam(detector, initial_objects)
    elif args.input:
        if not args.objects:
            logger.error("Objects parameter is required for image/video processing")
            return

        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file does not exist: {args.input}")
            return

        # Determine if input is image or video
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

        if input_path.suffix.lower() in image_extensions:
            process_image(detector, args.input, initial_objects)
        elif input_path.suffix.lower() in video_extensions:
            process_video(detector, args.input, initial_objects)
        else:
            logger.error(f"Unsupported file format: {input_path.suffix}")
    else:
        logger.error("Please specify either --input or --webcam")


if __name__ == "__main__":
    main()
