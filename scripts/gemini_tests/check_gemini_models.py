"""Check available Google Gemini models"""
import google.generativeai as genai
import os
from autoprepml.config_manager import AutoPrepMLConfig

# Get API key
api_key = AutoPrepMLConfig.get_api_key('google')
genai.configure(api_key=api_key)

print("Available Gemini Models:")
print("=" * 80)

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"✅ {model.name}")
        print(f"   Display Name: {model.display_name}")
        print(f"   Description: {model.description}")
        print()
