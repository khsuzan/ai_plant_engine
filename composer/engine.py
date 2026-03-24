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
### ROLE
You are a high-precision Digital Image Compositor.

### TASK
Perform a non-destructive insertion of specific plant assets onto a provided base garden image. 

### CONSTRAINTS
1.  **Zero-Change Policy:** The base garden image (lighting, geometry, background, and existing textures) must remain 100% identical. Do not re-render or "hallucinate" changes to the environment.
2.  **Precise Placement:** Map the (x, y) coordinates exactly. 
    * x: 0 (Left) to 1 (Right)
    * y: 0 (Top/Horizon) to 1 (Bottom/Foreground)
3.  **Perspective & Scaling:** Adjust the plant size based on the `scale` property, ensuring it obeys the depth of the garden (objects further "up" the y-axis should appear smaller to match the camera's FOV).
4.  **Integration (The "Blend"):** * Cast contact shadows on the ground directly beneath the inserted plants.
    * Match the plant’s highlights and shadows to the existing light source in the garden image.
    * Ensure the "seams" where the plant meets the soil are naturally occluded.

### INPUT DATA
- **Base Image:** {garden_image_url}
- **Plant Data:** {plants_block}

### FINAL OUTPUT
A single high-resolution photograph where the only delta between the input and output is the addition of the specified plants.
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
