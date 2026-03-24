import logging
import os
from typing import List, Dict, Any

from openai import OpenAI

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PlantComposer:
    """
    AI-powered plant image composer for garden projects.
    Uses OpenAI (gpt-image-1) for composition.
    """
    
    def __init__(self, **kwargs):
        self.api_key = (
            kwargs.get('openai_api_key') or 
            kwargs.get('api_key') or 
            os.getenv("OPENAI_API_KEY")
        )
        self.client = OpenAI(api_key=self.api_key)
        self.engine_id = "gpt-image-1"
    
    def build_garden_composition_prompt(self, garden_image_url: str, plants: List[Dict[str, Any]]) -> str:
        """
        Builds the text prompt instructing the AI to compose the garden image.
        """
        plant_descriptions = []
        for i, plant in enumerate(plants):
            img_src = plant.get('image_url') or plant.get('image', '')
            x = plant.get('x', 0.5)
            y = plant.get('y', 0.5)
            scale = plant.get('scale', 1.0)
            plant_descriptions.append(
                f"Plant {i+1}:\n"
                f"- image: {img_src}\n"
                f"- position: x={x}, y={y} (relative, 0 to 1)\n"
                f"- scale: {scale} (relative size)"
            )

        plants_block = "\n".join(plant_descriptions)

        prompt = f"""
Edit the provided garden image {garden_image_url} by inserting the specified plants while keeping everything else in the image completely unchanged.

CORE RULE
Do NOT modify, regenerate, or alter the background, lighting, textures, camera angle, or existing objects.
The output must be identical to the original image except for the added plants.

PLANT INSERTION RULES
Place each plant using normalized coordinates:
x: 0 = left edge, 1 = right edge
y: 0 = top/background, 1 = bottom/foreground
Position plants exactly according to the given coordinates.
Apply correct perspective scaling:
Objects closer to y = 1 should appear larger
Objects closer to y = 0 should appear smaller

LIGHTING & BLENDING
Match the original scene lighting direction and intensity.
Generate realistic contact shadows under each plant.
Ensure plants are naturally grounded and partially blended into the soil/terrain.
Match color tone and contrast to the environment.

PLANT DATA
{plants_block}

OUTPUT REQUIREMENT
Return a single high-resolution image where the only change from the original is the seamless insertion of the specified plants.
"""
        return prompt.strip()

    def compose_plants(
        self,
        garden_image: str,  
        plants: List[Dict[str, Any]],
        quality: str = 'standard',
        size: str = '1024x1024',
        blend_prompt: str = None,
        return_combined_only: bool = False
    ) -> Dict[str, Any]:
        """
        Main method to compose plants into the garden photo using OpenAI API.
        """
        logger.info(f"Starting plant composition with {len(plants)} plants using {self.engine_id}")
        
        prompt = self.build_garden_composition_prompt(garden_image, plants)
        
        if return_combined_only:
            return {
                'composite_image': None,
                'blended_image': None,
                'revised_prompt': prompt
            }
        
        logger.info(f"Sending to OpenAI for blending with {self.engine_id}...")
        
        try:
            # Send the text prompt to OpenAI Images API
            response = self.client.images.generate(
                model=self.engine_id,
                prompt=prompt,
                size=size,
            )
            
            blended_image = None
            revised_prompt = prompt
            
            if response.data and len(response.data) > 0:
                img_data = response.data[0]
                blended_image = getattr(img_data, 'b64_json', None) or getattr(img_data, 'url', None)
                if hasattr(img_data, 'revised_prompt') and img_data.revised_prompt:
                    revised_prompt = img_data.revised_prompt
            
            return {
                'composite_image': None,
                'blended_image': blended_image,
                'revised_prompt': revised_prompt
            }
        except Exception as e:
            logger.error(f"OpenAI API failed: {e}")
            raise

    def quick_blend(self, garden_image: str, plant_image: str, x: float = 0.5, y: float = 0.5, scale: float = 1.0) -> Dict[str, Any]:
        plants = [{'image': plant_image, 'x': x, 'y': y, 'scale': scale}]
        return self.compose_plants(garden_image, plants)
