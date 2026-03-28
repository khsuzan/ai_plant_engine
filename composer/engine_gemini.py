import os
import io
import logging
import google.generativeai as genai
from PIL import Image
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantComposerGemini:
    """
    AI Plant Composer utilizing Google's Gemini 2.5 Flash Image.
    """
    
    def __init__(self, **kwargs):
        self.api_key = (
            kwargs.get('gemini_api_key') or 
            kwargs.get('api_key') or 
            os.getenv("GEMINI_API_KEY")
        )
        self.model_name = "gemini-2.5-flash-image"

    def compose_plants(
        self, 
        garden_image_path: str, 
        plants: List[Dict[str, Any]], 
        size: str = "1024x1024"
    ) -> Dict[str, Any]:
        """
        Implementation adapted exactly from the provided Gemini script,
        conforming to the compose_plants signature found in engine.py.
        """
        if not self.api_key:
            return {"success": False, "error": "API Key not found. Please set GEMINI_API_KEY."}
            
        if not plants:
            return {"success": False, "error": "No plants provided."}

        # Configure Gemini
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)

        # The user's script takes the plant/flower image and transforms its background.
        # We extract the plant image path from the first plant.
        p = plants[0]
        plant_img_path = p.get('image_path') or p.get('image')
        
        logger.info(f"Loading image from: {plant_img_path}")
        
        try:
            input_image = Image.open(plant_img_path)
        except FileNotFoundError:
            return {"success": False, "error": f"Error: The file {plant_img_path} was not found."}
        except Exception as e:
            return {"success": False, "error": f"Error loading image: {str(e)}"}

        prompt = """
        Transform the flower in this image into a realistic garden-planted scene.

        Strict Guidelines:
        1. ISOLATE the flower or plant from the original image cleanly.
        2. PLACE the flower naturally into a garden bed or soil, as if it is growing there.
        3. BACKGROUND should be a clean, aesthetically pleasing garden environment (lush greenery, soft soil, minimal distractions).
        4. LIGHTING should be soft, natural daylight with even illumination to highlight the flower’s texture and color accurately.
        5. PRESERVE the original flower’s pattern, petal structure, texture, and color exactly.
        6. Ensure the plant is upright, properly rooted in the soil, and positioned naturally (no floating or artificial placement).
        7. Add subtle environmental elements like leaves, grass, or blurred background plants to enhance realism without overpowering the subject.
        8. Depth of field should be slightly shallow for a professional photography look (sharp focus on the flower, softly blurred background).

        Output ONLY the image.
        """

        logger.info("Sending request to Gemini... (this may take a few seconds)")

        try:
            # We pass the PIL Image object directly to the model
            response = model.generate_content([prompt, input_image])

            if response.parts:
                for part in response.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        # Found image data
                        img_data = part.inline_data.data
                        preview_image = Image.open(io.BytesIO(img_data))
                        
                        logger.info("✅ Success! Image generated.")
                        return {
                            "success": True,
                            "final_image_bytes": img_data,
                            "preview_image": preview_image
                        }

            # Handle cases where the model returns text (often a refusal or error explanation)
            if response.text:
                error_msg = f"Transformation failed. The model returned text instead of an image:\n{response.text}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}

            return {"success": False, "error": "No expected data returned from model."}

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return {"success": False, "error": str(e)}