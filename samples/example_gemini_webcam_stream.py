"""
Gemini Webcam Stream Processor
Real-time webcam processing with Google's Gemini AI model.
"""

import io
import json
import os
import threading
import time
import warnings

import cv2
import google.generativeai as genai
from dotenv import load_dotenv
from gemini_prompt import SYSTEM_PROMPT
from PIL import Image

# Configuration
warnings.filterwarnings("ignore", module="requests")
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# Constants
FRAME_PROCESS_INTERVAL = 0.5  # seconds
TASK_MONITOR_INTERVAL = 0.5  # seconds
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
COMPLETION_KEYWORDS = ["task completed", "task finished", "task done", "completed successfully"]


class GeminiWebcamProcessor:
    """Manages webcam processing state and Gemini AI interactions."""

    def __init__(self):
        self.current_task = ""
        self.current_response = ""
        self.processing = False
        self.task_completed = False

        # Thread synchronization
        self.task_lock = threading.Lock()
        self.response_lock = threading.Lock()

        # FPS tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0

    def update_task(self, task: str) -> None:
        """Update the current task and reset completion status."""
        with self.task_lock:
            self.current_task = task
            self.task_completed = False

    def get_task(self) -> str:
        """Get the current task."""
        with self.task_lock:
            return self.current_task

    def clear_task(self) -> None:
        """Clear the current task and mark as completed."""
        with self.task_lock:
            self.current_task = ""
            self.task_completed = True

    def is_task_completed(self) -> bool:
        """Check if the current task is completed."""
        with self.task_lock:
            return self.task_completed

    def update_response(self, response: str) -> None:
        """Update the AI response and set processing to false."""
        with self.response_lock:
            self.current_response = response
            self.processing = False

    def get_response(self) -> tuple[str, bool]:
        """Get the current response and processing status."""
        with self.response_lock:
            return self.current_response, self.processing

    def set_processing(self, status: bool) -> None:
        """Set the processing status."""
        with self.response_lock:
            self.processing = status

    def update_fps(self) -> None:
        """Update FPS calculation."""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            self.fps = self.frame_count / elapsed_time

    def get_fps(self) -> float:
        """Get current FPS."""
        return self.fps


def _convert_frame_to_image(frame) -> bytes:
    """Convert OpenCV frame to JPEG bytes for Gemini upload."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return img_byte_arr


def _check_task_completion(response_text: str) -> bool:
    """Check if the task is completed based on response text."""
    try:
        response_data = json.loads(response_text)
        return response_data.get("task_completed", False)
    except json.JSONDecodeError:
        # Check for completion keywords in plain text response
        return any(keyword in response_text.lower() for keyword in COMPLETION_KEYWORDS)


def process_frame_with_gemini(processor: GeminiWebcamProcessor, frame) -> None:
    """Process frame with Gemini AI in separate thread."""
    task = processor.get_task()
    if not task or processor.processing:
        return

    processor.set_processing(True)

    try:
        # Convert frame to format suitable for Gemini
        img_byte_arr = _convert_frame_to_image(frame)
        image_part = genai.upload_file(img_byte_arr, mime_type="image/jpeg")

        # Generate AI response
        prompt = [SYSTEM_PROMPT.format(task=task), image_part]
        response = model.generate_content(prompt)

        processor.update_response(response.text)

        # Check for task completion
        if _check_task_completion(response.text):
            print(f"\n✅ Task '{task}' completed! Clearing task.")
            processor.clear_task()

        print(f"\nTask: {task}")
        print(f"Response: {response.text}")

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        processor.update_response(error_msg)
        print(f"Error processing: {e}")


def input_handler(processor: GeminiWebcamProcessor) -> None:
    """Handle user input in separate thread."""
    print("Enter task: ", end="", flush=True)

    while True:
        try:
            task = input().strip()
            if task:
                processor.update_task(task)
                print(f"✅ Task updated: {task}")
            print("Enter task: ", end="", flush=True)
        except (EOFError, KeyboardInterrupt):
            break


def task_completion_monitor(processor: GeminiWebcamProcessor) -> None:
    """Monitor for task completion and prompt for new tasks."""
    last_completion_status = False

    while True:
        try:
            current_completion_status = processor.is_task_completed()

            # If task just completed (transition from not completed to completed)
            if current_completion_status and not last_completion_status and processor.get_task():

                time.sleep(1)  # Wait for any final processing
                print("\n🔄 Task completed! Enter a new task: ", end="", flush=True)

            last_completion_status = current_completion_status
            time.sleep(TASK_MONITOR_INTERVAL)

        except Exception as e:
            print(f"Task monitor error: {e}")
            time.sleep(1)


def _draw_text_with_background(
    frame,
    text: str,
    position: tuple,
    font_scale: float = 0.6,
    color: tuple = (255, 255, 255),
    thickness: int = 2,
) -> None:
    """Draw text with a semi-transparent background for better visibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Draw background rectangle
    x, y = position
    cv2.rectangle(
        frame, (x - 5, y - text_height - 5), (x + text_width + 5, y + baseline + 5), (0, 0, 0), -1
    )

    # Draw text
    cv2.putText(frame, text, position, font, font_scale, color, thickness)


def _display_json_response(frame, response_data: dict, y_offset: int) -> int:
    """Display structured JSON response data on frame."""
    current_y = y_offset

    # Display current state
    if "current_state" in response_data:
        current_state = response_data["current_state"][:50]  # Truncate if too long
        _draw_text_with_background(
            frame, f"Status: {current_state}", (10, current_y), color=(255, 255, 0)
        )
        current_y += 25

    # Display completion status or next instruction
    if response_data.get("task_completed", False):
        _draw_text_with_background(frame, "TASK COMPLETED!", (10, current_y), color=(0, 255, 0))
    elif "current_instruction" in response_data:
        instruction = response_data["current_instruction"][:50]
        _draw_text_with_background(
            frame, f"Next: {instruction}", (10, current_y), color=(0, 255, 255)
        )

    return current_y + 25


def _display_text_response(frame, response: str, y_offset: int) -> None:
    """Display plain text response on frame."""
    lines = response.split("\n")[:3]  # Show first 3 lines
    for i, line in enumerate(lines):
        if line.strip():  # Skip empty lines
            truncated_line = line[:60]  # Truncate long lines
            _draw_text_with_background(
                frame,
                truncated_line,
                (10, y_offset + i * 25),
                font_scale=0.5,
                color=(255, 255, 255),
            )


def draw_overlay(frame, processor: GeminiWebcamProcessor) -> None:
    """Draw task and response overlay on frame."""
    task = processor.get_task()
    response, processing = processor.get_response()

    current_y = 30

    # Display FPS
    fps = processor.get_fps()
    _draw_text_with_background(
        frame, f"FPS: {fps:.1f}", (10, current_y), font_scale=0.7, color=(0, 255, 0)
    )
    current_y += 30

    # Display current task
    if task:
        task_status = " (COMPLETED)" if processor.is_task_completed() else ""
        task_color = (0, 255, 0) if processor.is_task_completed() else (255, 255, 255)
        truncated_task = task[:40] + "..." if len(task) > 40 else task

        _draw_text_with_background(
            frame, f"Task: {truncated_task}{task_status}", (10, current_y), color=task_color
        )
        current_y += 30

    # Display processing status or response
    if processing:
        _draw_text_with_background(frame, "Processing...", (10, current_y), color=(0, 255, 255))
    elif response:
        try:
            response_data = json.loads(response)
            _display_json_response(frame, response_data, current_y)
        except json.JSONDecodeError:
            _display_text_response(frame, response, current_y)


def _initialize_webcam() -> cv2.VideoCapture:
    """Initialize and configure the webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open webcam")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def _start_background_threads(processor: GeminiWebcamProcessor) -> None:
    """Start input handler and task completion monitor threads."""
    input_thread = threading.Thread(target=input_handler, args=(processor,), daemon=True)
    input_thread.start()

    monitor_thread = threading.Thread(
        target=task_completion_monitor, args=(processor,), daemon=True
    )
    monitor_thread.start()


def _should_process_frame(
    current_time: float, last_process_time: float, processor: GeminiWebcamProcessor
) -> bool:
    """Determine if frame should be processed based on timing and task status."""
    return (
        current_time - last_process_time >= FRAME_PROCESS_INTERVAL
        and processor.get_task()
        and not processor.processing
    )


def main() -> None:
    """Main application entry point."""
    processor = GeminiWebcamProcessor()

    try:
        # Initialize webcam
        cap = _initialize_webcam()
        print("🎥 Gemini Webcam started. Press 'q' in video window to quit.")

        # Start background threads
        _start_background_threads(processor)

        last_process_time = 0

        # Main processing loop
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame from webcam")
                continue

            # Update FPS calculation
            processor.update_fps()

            # Process frame at regular intervals
            current_time = time.time()
            if _should_process_frame(current_time, last_process_time, processor):
                threading.Thread(
                    target=process_frame_with_gemini, args=(processor, frame.copy()), daemon=True
                ).start()
                last_process_time = current_time

            # Draw overlay and display frame
            draw_overlay(frame, processor)
            cv2.imshow("Gemini Webcam", frame)

            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Shutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        if "cap" in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("Application exited cleanly.")


if __name__ == "__main__":
    main()
