import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

print("Available Models:")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
