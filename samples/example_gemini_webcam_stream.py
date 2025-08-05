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

warnings.filterwarnings("ignore", module="requests")  # Ignore warnings from requests library
# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")


class GeminiWebcamProcessor:
    def __init__(self):
        self.current_task = ""
        self.current_response = ""
        self.processing = False
        self.task_completed = False
        self.task_lock = threading.Lock()
        self.response_lock = threading.Lock()

    def update_task(self, task):
        with self.task_lock:
            self.current_task = task
            self.task_completed = False  # Reset completion status

    def get_task(self):
        with self.task_lock:
            return self.current_task

    def clear_task(self):
        with self.task_lock:
            self.current_task = ""
            self.task_completed = True

    def is_task_completed(self):
        with self.task_lock:
            return self.task_completed

    def update_response(self, response):
        with self.response_lock:
            self.current_response = response
            self.processing = False

    def get_response(self):
        with self.response_lock:
            return self.current_response, self.processing

    def set_processing(self, status):
        with self.response_lock:
            self.processing = status


def process_frame_with_gemini(processor, frame):
    """Process frame with Gemini in separate thread"""
    task = processor.get_task()
    if not task or processor.processing:
        return

    processor.set_processing(True)

    try:
        # Convert frame to PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Convert to bytes for upload
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)

        # Upload to Gemini
        image_part = genai.upload_file(img_byte_arr, mime_type="image/jpeg")

        # Create prompt and get response
        prompt = [SYSTEM_PROMPT.format(task=task), image_part]
        response = model.generate_content(prompt)

        processor.update_response(response.text)

        # Check if task is completed
        try:
            response_data = json.loads(response.text)
            if "task_completed" in response_data and response_data["task_completed"]:
                print(f"Task '{task}' completed! Clearing task.")
                processor.clear_task()
        except json.JSONDecodeError:
            # Check for completion keywords in plain text response
            completion_keywords = [
                "task completed",
                "task finished",
                "task done",
                "completed successfully",
            ]
            if any(keyword in response.text.lower() for keyword in completion_keywords):
                print(f"Task '{task}' completed! Clearing task.")
                processor.clear_task()

        print(f"\nTask: {task}")
        print(f"Response: {response.text[:200]}...")

    except Exception as e:
        processor.update_response(f"Error: {str(e)}")
        print(f"Error processing: {e}")


def input_handler(processor):
    """Handle user input in separate thread"""
    print("Enter task (or 'quit'): ", end="", flush=True)
    while True:
        try:
            task = input().strip()
            if task.lower() in ["quit", "q"]:
                break
            if task:
                processor.update_task(task)
                print(f"Task updated: {task}")
            print("Enter task (or 'quit'): ", end="", flush=True)
        except (EOFError, KeyboardInterrupt):
            break


def draw_overlay(frame, processor):
    """Draw task and response overlay on frame"""
    task = processor.get_task()
    response, processing = processor.get_response()

    # Add timestamp
    timestamp = time.strftime("%H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Add task
    if task:
        task_status = " (COMPLETED)" if processor.is_task_completed() else ""
        cv2.putText(
            frame,
            f"Task: {task[:50]}{task_status}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if processor.is_task_completed() else (255, 255, 255),
            2,
        )

    # Add processing status or response
    if processing:
        cv2.putText(
            frame, "Processing...", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
    elif response:
        try:
            # Try to parse JSON response
            response_data = json.loads(response)

            # Display current state
            if "current_state" in response_data:
                current_state = response_data["current_state"][:80]
                cv2.putText(
                    frame,
                    f"Status: {current_state}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

            # Display current instruction or completion status
            if "task_completed" in response_data and response_data["task_completed"]:
                cv2.putText(
                    frame,
                    "TASK COMPLETED!",
                    (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            elif "current_instruction" in response_data:
                current_instruction = response_data["current_instruction"][:80]
                cv2.putText(
                    frame,
                    f"Next: {current_instruction}",
                    (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )

        except json.JSONDecodeError:
            # Fallback to showing first 3 lines if not valid JSON
            lines = response.split("\n")[:3]
            for i, line in enumerate(lines):
                cv2.putText(
                    frame,
                    line[:60],
                    (10, 90 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )


def main():
    processor = GeminiWebcamProcessor()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("Gemini Webcam started. Press 'q' in video window to quit.")

    # Start input thread
    input_thread = threading.Thread(target=input_handler, args=(processor,), daemon=True)
    input_thread.start()

    last_process_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame every 1 second
            current_time = time.time()
            if current_time - last_process_time >= 1.0 and processor.get_task():
                threading.Thread(
                    target=process_frame_with_gemini, args=(processor, frame.copy()), daemon=True
                ).start()
                last_process_time = current_time

            # Draw overlay
            draw_overlay(frame, processor)

            # Display frame
            cv2.imshow("Gemini Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Exited.")


if __name__ == "__main__":
    main()
