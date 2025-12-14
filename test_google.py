"""
This is a simple test to ensure that the Google Generative AI SDK is correctly configured 
and can list available models.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env file
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print(" Error: No API Key found in .env")
else:
    print(f" Found API Key: {api_key[:5]}...")
    
    # Configure the Google SDK
    genai.configure(api_key=api_key)

    print("\nAttempting to list available models...")
    try:
        # Loop through all models and print only the ones that generate text
        found_any = False
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f" - {m.name}")
                found_any = True
        
        if not found_any:
            print(" No text generation models found. Check your API Key permissions.")
            
    except Exception as e:
        print(f"\n connection error:\n{e}")