import logging
import os
import io
import base64
import requests
from typing import List, Dict, Any

# Image processing
from PIL import Image

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PlantComposer:
    """
    AI-powered plant image composer that blends multiple plants into a garden image.
    Uses Hugging Face's Free Inference API (Stable Diffusion) for true Image-to-Image blending.
    """
    
    def __init__(self, huggingface_api_key: str):
        self.api_key = huggingface_api_key
        # Using standard Stable Diffusion 1.5 - reliable and supported by the free API
        self.api_url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    
    def compose_plants(
        self,
        garden_image: str,  
        plants: List[Dict[str, Any]],
        blend_prompt: str = None,
        return_combined_only: bool = False
    ) -> Dict[str, Any]:
        """
        Main method to compose plants into garden and blend them.
        """
        logger.info(f"Starting plant composition with {len(plants)} plants")
        
        # Step 1: Load and composite the garden + plants
        composite_base64 = self._create_composite(garden_image, plants)
        logger.info("Created initial composite image")
        
        if return_combined_only:
            return {
                'composite_image': composite_base64,
                'blended_image': None,
                'revised_prompt': None
            }
        
        # Step 2: Generate blend prompt if not provided
        if blend_prompt is None:
            blend_prompt = self._generate_blend_prompt(plants)
        
        # Step 3: Send to Hugging Face for natural blending
        logger.info("Sending to Hugging Face for AI blending. (This may take 20s if the free model is waking up...)")
        blended_result = self._blend_with_huggingface(
            composite_image=composite_base64,
            prompt=blend_prompt
        )
        
        return {
            'composite_image': composite_base64,
            'blended_image': blended_result['image_base64'],
            'revised_prompt': blended_result.get('revised_prompt', blend_prompt)
        }
    
    def _load_image(self, image_source: str) -> Image.Image:
        """Load an image from URL, base64, or file path. Bypasses 403 Forbidden errors."""
        try:
            if image_source.startswith('data:image'):
                # Base64 image
                header, b64data = image_source.split(',', 1)
                img_data = base64.b64decode(b64data)
                return Image.open(io.BytesIO(img_data)).convert('RGBA')
            
            elif image_source.startswith('http://') or image_source.startswith('https://'):
                # URL image - Using requests with a User-Agent to prevent Wikipedia 403 blocks
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
                }
                response = requests.get(image_source, headers=headers, timeout=10)
                response.raise_for_status() 
                return Image.open(io.BytesIO(response.content)).convert('RGBA')
            
            else:
                # File path
                return Image.open(image_source).convert('RGBA')
                
        except Exception as e:
            logger.error(f"Failed to load image from {image_source[:50]}...: {e}")
            raise ValueError(f"Could not load image: {e}")
    
    def _create_composite(
        self, 
        garden_image: str, 
        plants: List[Dict[str, Any]]
    ) -> str:
        """
        Composite plants onto garden image at specified positions.
        Returns base64 of the composite image.
        """
        garden = self._load_image(garden_image)
        original_width, original_height = garden.size
        
        logger.info(f"Garden image size: {original_width}x{original_height}")
        
        output = Image.new('RGBA', (original_width, original_height), (0, 0, 0, 0))
        output.paste(garden, (0, 0))
        
        for i, plant in enumerate(plants):
            try:
                plant_img = self._load_image(plant['image'])
                
                x = plant.get('x', 0.5)
                y = plant.get('y', 0.5)
                
                if x <= 1 and y <= 1:
                    x = int(x * original_width)
                    y = int(y * original_height)
                
                scale = plant.get('scale', 1.0)
                if scale != 1.0:
                    new_width = int(plant_img.width * scale)
                    new_height = int(plant_img.height * scale)
                    plant_img = plant_img.resize((new_width, new_height), Image.LANCZOS)
                
                plant_width, plant_height = plant_img.size
                paste_x = x - plant_width // 2
                paste_y = y - plant_height // 2
                
                output.paste(plant_img, (paste_x, paste_y), plant_img)
                logger.info(f"Planted plant {i+1} at ({paste_x}, {paste_y})")
                
            except Exception as e:
                logger.warning(f"Failed to place plant {i+1}: {e}")
                continue
        
        output_rgb = output.convert('RGB')
        
        # Save to bytes as JPEG for Hugging Face
        buffer = io.BytesIO()
        output_rgb.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def _generate_blend_prompt(self, plants: List[Dict[str, Any]]) -> str:
        """Generate a prompt for AI to blend plants naturally."""
        plant_descriptions = []
        for plant in plants:
            desc = plant.get('description', 'a beautiful plant')
            if plant.get('name'):
                desc = f"{plant['name']} - {desc}"
            plant_descriptions.append(desc)
        
        plants_text = ", ".join(plant_descriptions) if plant_descriptions else "plants and landscaping"
        
        return (
            f"A photorealistic backyard garden featuring {plants_text}. "
            f"Perfect daytime lighting, natural shadows on the ground, "
            f"seamless landscaping integration, highly detailed, 8k resolution, photorealistic."
        )
    
    def _blend_with_huggingface(
        self,
        composite_image: str,
        prompt: str,
    ) -> Dict[str, Any]:
        """
        Send composite image to Hugging Face Free Inference API.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": composite_image,  # HF accepts the base64 string directly
            "parameters": {
                "prompt": prompt,
                "negative_prompt": "blurry, unnatural, morphed shapes, floating objects",
                # 0.45 strength means: Keep 55% of the original image, change 45% (to blend lighting/shadows)
                "strength": 0.45, 
                "guidance_scale": 7.5
            }
        }
        
        response = requests.post(self.api_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            logger.error(f"Hugging Face API failed: {response.text}")
            raise Exception(f"Non-200 response: {response.status_code} - {response.text}")
            
        # Hugging face returns the raw image bytes in the response content
        image_data = response.content
        final_image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        logger.info("AI blending complete!")
        
        return {
            'image_base64': final_image_base64,
            'revised_prompt': prompt,
        }

    def quick_blend(self, garden_image: str, plant_image: str, x: float = 0.5, y: float = 0.5, scale: float = 1.0) -> Dict[str, Any]:
        plants = [{'image': plant_image, 'x': x, 'y': y, 'scale': scale}]
        return self.compose_plants(garden_image, plants)


# Standalone testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # NOTE: Change this to look for HUGGINGFACE_API_KEY
    api_key = os.environ.get("HUGGINGFACE_API_KEY")
    if not api_key:
        print("ERROR: HUGGINGFACE_API_KEY not found in environment")
        exit(1)
    
    print("Initializing PlantComposer with Hugging Face...")
    composer = PlantComposer(huggingface_api_key=api_key)
    
    print("\nComposer is ready to test!")