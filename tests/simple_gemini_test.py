"""Simple Gemini API test with basic prompt"""
import google.generativeai as genai
from autoprepml.config_manager import AutoPrepMLConfig

# Get API key
api_key = AutoPrepMLConfig.get_api_key('google')
genai.configure(api_key=api_key)

# Create model
model = genai.GenerativeModel('gemini-2.5-flash')

# Simple test
print("Testing Gemini API with simple prompt...")
print("=" * 80)

try:
    response = model.generate_content(
        "Explain what data imputation means in simple terms.",
        safety_settings={
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
        }
    )
    
    print("\n✅ Response received:")
    print(response.text)
    print("\n" + "=" * 80)
    print("✅ Gemini API is working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
