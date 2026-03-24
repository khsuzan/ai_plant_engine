import logging
import os
import io
import base64
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFilter
from openai import OpenAI

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlantComposer:
    """
    Surgical AI Plant Composer.
    Guarantees background preservation by manually compositing and 
    using AI Inpainting for lighting and shadow grounding.
    """
    
    def __init__(self, **kwargs):
        self.api_key = (
            kwargs.get('openai_api_key') or 
            kwargs.get('api_key') or 
            os.getenv("OPENAI_API_KEY")
        )
        self.client = OpenAI(api_key=self.api_key)
        # Using the 2026 'gpt-image-1' for high-fidelity inpainting
        self.model_id = "gpt-image-1" 

    def _get_perspective_scale(self, y: float) -> float:
        """
        Calculates a multiplier based on vertical position.
        y=0 (horizon) -> 0.4x scale
        y=1 (foreground) -> 1.0x scale
        """
        min_scale = 0.4
        return min_scale + (y * (1.0 - min_scale))

    def _prepare_mask(self, base_size: tuple, plants_metadata: List[Dict]) -> Image.Image:
        """
        Creates a mask where White (255) is the area for the AI to 'heal'
        and Black (0) is the area to keep 100% original.
        """
        mask = Image.new("L", base_size, 0)
        draw = ImageDraw.Draw(mask)
        
        for p in plants_metadata:
            # We create a mask slightly larger than the plant to allow for shadow generation
            cx, cy = p['x_px'], p['y_px']
            radius = p['width'] * 0.7 
            
            # Draw a soft-edged circle at the base of the plant
            bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
            draw.ellipse(bbox, fill=255)
            
        return mask.filter(ImageFilter.GaussianBlur(radius=15))

    def _to_bytes(self, img: Image.Image) -> io.BytesIO:
        """Utility to convert PIL image to OpenAI-compatible byte stream."""
        byte_stream = io.BytesIO()
        img.save(byte_stream, format="PNG")
        byte_stream.seek(0)
        return byte_stream

    def compose_plants(
        self, 
        garden_image_path: str, 
        plants: List[Dict[str, Any]], 
        size: str = "1024x1024"
    ) -> Dict[str, Any]:
        """
        Performs the full composition: Manual Paste -> Masking -> AI Refinement.
        """
        logger.info(f"Processing {len(plants)} plants for surgical insertion.")
        
        try:
            # 1. Load background
            garden = Image.open(garden_image_path).convert("RGBA")
            canvas = garden.copy()
            plants_metadata = []

            # 2. Manual Perspective Composition
            for p in plants:
                plant_img = Image.open(p['image_path']).convert("RGBA")
                
                # Apply scaling logic
                p_scale = self._get_perspective_scale(p['y'])
                total_scale = p.get('scale', 1.0) * p_scale
                
                new_size = (int(plant_img.width * total_scale), int(plant_img.height * total_scale))
                plant_img = plant_img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Calculate coordinates (anchoring the bottom-center of the plant)
                x_pos = int(p['x'] * garden.width - (plant_img.width / 2))
                y_pos = int(p['y'] * garden.height - plant_img.height)
                
                canvas.alpha_composite(plant_img, (x_pos, y_pos))
                
                # Store metadata for masking
                plants_metadata.append({
                    'x_px': x_pos + (plant_img.width // 2),
                    'y_px': y_pos + plant_img.height, # Focus mask on the ground/base
                    'width': plant_img.width
                })

            # 3. Generate Inpainting Mask
            mask = self._prepare_mask(garden.size, plants_metadata)
            
            # 4. Request AI Refinement (The 'Weld')
            # This fixes the lighting and adds the shadows onto the original soil.
            prompt = (
                "Photorealistic garden edit. Ground the inserted plants into the soil "
                "with soft contact shadows. Match the ambient lighting and color "
                "temperature exactly. Keep all unmasked areas identical."
            )

            logger.info("Sending surgical edit request to OpenAI...")
            response = self.client.images.edit(
                model=self.model_id,
                image=self._to_bytes(canvas.convert("RGB")),
                mask=self._to_bytes(mask),
                prompt=prompt,
                size=size,
            )

            return {
                "success": True,
                "final_image_url": response.data[0].url,
                "preview_image": canvas # You can use this for instant UI feedback
            }

        except Exception as e:
            logger.error(f"Failed to compose: {str(e)}")
            return {"success": False, "error": str(e)}

# --- Example Usage ---
# composer = PlantComposer()
# result = composer.compose_plants(
#     garden_image_path="my_backyard.png",
#     plants=[
#         {"image_path": "lavender.png", "x": 0.5, "y": 0.8, "scale": 1.2, "species": "Lavender"},
#         {"image_path": "rose.png", "x": 0.2, "y": 0.5, "scale": 0.8, "species": "Red Rose"}
#     ]
# )