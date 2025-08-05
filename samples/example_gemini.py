import os

import google.generativeai as genai
from dotenv import load_dotenv
from gemini_prompt import SYSTEM_PROMPT

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Configure the API with the key from environment
genai.configure(api_key=API_KEY)

# Load the model
model = genai.GenerativeModel("gemini-1.5-flash")

# Load an image (you can use a local file or a URI)
image_path = "sample.png"
image_part = genai.upload_file(image_path)

# Create the prompt with both text and the image
task = "boil the egg"
prompt = [SYSTEM_PROMPT.format(task=task), image_part]

# Generate content
response = model.generate_content(prompt)

# Print the model's response
print(response.text)
