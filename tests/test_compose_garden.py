import os
import sys
import base64
import unittest
from io import BytesIO
from PIL import Image

# Add the project root to sys.path so we can import 'composer' locally without Django
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from composer.engine import PlantComposer

def load_and_prepare_image(filepath, is_plant=False):
    """
    Loads a local image and converts it to base64 simply and plainly.
    We are no longer auto-resizing the garden here, because the ENGINE 
    is now correctly configured to auto-fit it to SDXL constraints.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing required test image: {filepath}")

    img = Image.open(filepath)
    
    if is_plant:
        img = img.convert('RGBA')
        img.thumbnail((300, 300), Image.Resampling.LANCZOS)
    else:
        img = img.convert('RGB')
        
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"


class TestPlantComposerReal(unittest.TestCase):
    def setUp(self):
        self.api_key = os.environ.get('STABILITY_API_KEY')
        if not self.api_key:
            self.skipTest("STABILITY_API_KEY env var missing.")
            
        self.composer = PlantComposer(stability_api_key=self.api_key)

    def test_compose_plants_real_user_images(self):
        garden_path = "garden.jpg"   # The garden landscape
        tomato_path = "tomato.jpg"   # The tomato plant
        
        # Load the raw user provided sizes
        # The engine will auto-scale the garden to closest acceptable SDXL spec
        garden_b64 = load_and_prepare_image(garden_path)
        plant_b64 = load_and_prepare_image(tomato_path, is_plant=True)

        plants_input = [
            {
                "image": plant_b64,
                "x": 0.5, # Bottom-mid placement
                "y": 0.5, 
                "scale": 1.0,
                "name": "a lush red tomato plant"
            }
        ]

        print("\n" + "="*50)
        print("Sending REAL custom images to Stability AI API...")
        print("="*50)

        # Run the composer function
        result = self.composer.compose_plants(
            garden_image=garden_b64,
            plants=plants_input
        )

        # Saves tracking images
        if result.get("composite_image"):
            composite_str = result["composite_image"].split(',', 1)[1] if ',' in result["composite_image"] else result["composite_image"]
            with open("test_real_user_composite.png", "wb") as f:
                f.write(base64.b64decode(composite_str))
            print("✅ Saved PRE-AI COMPOSITE to: test_real_user_composite.png")

        if result.get("blended_image"):
            blended_str = result["blended_image"].split(',', 1)[1] if ',' in result["blended_image"] else result["blended_image"]
            with open("test_real_user_blended.png", "wb") as f:
                f.write(base64.b64decode(blended_str))
            print("✅ Saved FINAL AI BLENDED RESULT to: test_real_user_blended.png")

if __name__ == '__main__':
    unittest.main()
