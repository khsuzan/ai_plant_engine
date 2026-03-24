import os
import sys

def start():
    """Start the Django local dummy server."""
    print("Starting Uvicorn ASGI Server...")
    os.system("uv run uvicorn dummy_project.asgi:application --reload")

def harvest():
    """Run the Django management command to harvest plant data."""
    print("Running Harvester Pipeline via Django...")
    args = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "--count 5"
    os.system(f"uv run python manage.py run_harvester {args}")

def test_engine():
    """Run the standalone bare python script for the Harvester Engine."""
    print("Running Standalone Python Engine (No Django)...")
    
    # If arguments are passed directly, use them
    if len(sys.argv) > 2:
        args = " ".join(sys.argv[2:])
        os.system(f"uv run python harvester/engine.py {args}")
        return
        
    # Otherwise, ask interactively
    while True:
        print("\n" + "="*50)
        print("🌱 Plant Harvester Interactive Test 🌱")
        print("="*50)
        plant_name = input("\nEnter a plant name to test (or 'q' to quit, 'random' for a random test): ")
        
        if plant_name.lower() in ['q', 'quit', 'exit']:
            print("Exiting...")
            break
            
        if plant_name.lower() == 'random':
            print("\nGenerating random plant...")
            os.system("uv run python harvester/engine.py")
        elif plant_name.strip():
            print(f"\nProcessing '{plant_name}'...")
            # We explicitly pass the properly quoted plant name to the engine
            os.system(f'uv run python harvester/engine.py "{plant_name}"')
        else:
            print("Please enter a valid plant name.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run scripts.py [start|harvest|test-engine]")
        # Default to interactive test if no args provided to make it friendly
        print("\nNo command provided. Defaulting to interactive test-engine mode...")
        test_engine()
        sys.exit(0)
        
    command = sys.argv[1]
    
    if command == "start":
        start()
    elif command == "harvest":
        harvest()
    elif command == "test-engine":
        test_engine()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: start, harvest, test-engine")
