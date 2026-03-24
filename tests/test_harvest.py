import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from harvester.engine import HarvestEngine


# Add project root to sys.path so we can import 'harvester' organically
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

def run_tests():
    load_dotenv()
    
    # Load keys
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if not openai_key or "your-" in openai_key:
        print("❌ Authentication Error: Please add your real OPENAI_API_KEY to the .env file in the root directory.")
        sys.exit(1)
        
    print("====================================")
    print("🌱 Initializing HarvestEngine Test...")
    print("====================================\n")
    
    engine = HarvestEngine(openai_api_key=openai_key)
    
    plant_to_test = "ZZ Plant"
    
    print(f"🔍 Testing Search & Knowledge Extraction for: '{plant_to_test}'...")
    print("This will trigger OpenAI structure generation.\n")
    
    try:
        result = engine._generate_single_plant(plant_to_test)
        print("\n✅ Successfully generated data! Here is the JSON output:\n")
        print(json.dumps(result, indent=2))
        print("\n🎉 Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")

if __name__ == "__main__":
    run_tests()
