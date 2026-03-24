import os
from openai import OpenAI
import traceback

client = OpenAI()
try:
    response = client.images.generate(
        model="dall-e-3",
        prompt="A cute cat",
        n=1,
        size="1024x1024",
        response_format="b64_json"
    )
    print("Success with b64_json")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
