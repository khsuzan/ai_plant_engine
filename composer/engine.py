import logging
import os
import io
import base64
import requests
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFilter

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PlantComposer:
    """
    AI-powered plant image composer for garden projects.
    Uses Stability AI (Stable Diffusion XL) for true Image-to-Image lighting and shadow blending.
    """
    
    def __init__(self, **kwargs):
        self.api_key = (
            kwargs.get('stability_api_key') or 
            kwargs.get('huggingface_api_key') or 
            kwargs.get('openai_api_key') or 
            kwargs.get('api_key')
        )
        self.api_host = "https://api.stability.ai"
        
        # WE ARE SWITCHING FROM IMAGE-TO-IMAGE TO PURE INPAINTING
        # Inpainting explicitly tells the AI to NEVER TOUCH the unmasked pixels.
        self.engine_id = "stable-diffusion-xl-1024-v1-0"
    
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
        Main method to compose plants into the garden photo and blend them via mask inpainting.
        """
        logger.info(f"Starting INPAINTING plant composition with {len(plants)} plants")
        
        # Step 1: Composite the garden + plants using Pillow, BUT ALSO generate a mask for just the plant areas
        composite_base64, mask_base64 = self._create_composite_and_mask(garden_image, plants)
        
        if return_combined_only:
            return {
                'composite_image': composite_base64,
                'blended_image': None,
                'revised_prompt': None
            }
        
        # Step 2: Generate blend prompt
        if blend_prompt is None:
            blend_prompt = self._generate_blend_prompt(plants)
        
        # Step 3: Send to Stability AI for INPAINTING (not img2img)
        logger.info("Sending to Stability AI for INPAINTING blending...")
        blended_result = self._blend_with_stability_inpainting(
            composite_image=composite_base64,
            mask_image=mask_base64,
            prompt=blend_prompt
        )
        
        return {
            'composite_image': composite_base64,
            'blended_image': blended_result['image_base64'],
            'revised_prompt': blended_result.get('revised_prompt', blend_prompt)
        }
    
    def _load_image(self, image_source: str) -> Image.Image:
        try:
            if image_source.startswith('data:image'):
                header, b64data = image_source.split(',', 1)
                img_data = base64.b64decode(b64data)
                return Image.open(io.BytesIO(img_data)).convert('RGBA')
            elif image_source.startswith('http://') or image_source.startswith('https://'):
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(image_source, headers=headers, timeout=10)
                response.raise_for_status() 
                return Image.open(io.BytesIO(response.content)).convert('RGBA')
            else:
                return Image.open(image_source).convert('RGBA')
        except Exception as e:
            logger.error(f"Failed to load image from {image_source[:50]}...: {e}")
            raise ValueError(f"Could not load image: {e}")
    
    def _create_composite_and_mask(self, garden_image: str, plants: List[Dict[str, Any]]) -> tuple[str, str]:
        """Composites plants and creates a strict black/white mask for Stability AI."""
        garden = self._load_image(garden_image)

        allowed_dimensions = [
            (1024, 1024), (1152, 896), (1216, 832), (1344, 768), 
            (1536, 640), (640, 1536), (768, 1344), (832, 1216), (896, 1152)
        ]
        orig_w, orig_h = garden.size
        orig_ratio = orig_w / orig_h
        
        best_dim = (1024, 1024)
        min_diff = float("inf")
        for dim_w, dim_h in allowed_dimensions:
            dim_ratio = dim_w / dim_h
            diff = abs(orig_ratio - dim_ratio)
            if diff < min_diff:
                min_diff = diff
                best_dim = (dim_w, dim_h)
        
        target_w, target_h = best_dim

        target_ratio = target_w / target_h
        if orig_ratio > target_ratio:
            new_w = int(target_ratio * orig_h)
            offset = (orig_w - new_w) / 2
            garden = garden.crop((offset, 0, orig_w - offset, orig_h))
        elif orig_ratio < target_ratio:
            new_h = int(orig_w / target_ratio)
            offset = (orig_h - new_h) / 2
            garden = garden.crop((0, offset, orig_w, orig_h - offset))
            
        garden = garden.resize((target_w, target_h), Image.Resampling.LANCZOS)

        output = Image.new('RGBA', (target_w, target_h), (0, 0, 0, 0))
        output.paste(garden, (0, 0))
        
        # MASK LOGIC: Black = Keep original garden, White = AI is allowed to change this part
        mask = Image.new('L', (target_w, target_h), 0)

        for i, plant in enumerate(plants):
            try:
                plant_img = self._load_image(plant['image'])
                x = plant.get('x', 0.5)
                y = plant.get('y', 0.5)
                
                if x <= 1 and y <= 1:
                    x = int(x * target_w)
                    y = int(y * target_h)
                
                scale = plant.get('scale', 1.0)
                if scale != 1.0:
                    new_width = int(plant_img.width * scale)
                    new_height = int(plant_img.height * scale)
                    plant_img = plant_img.resize((new_width, new_height), Image.LANCZOS)
                
                plant_width, plant_height = plant_img.size
                paste_x = int(x - plant_width // 2)
                paste_y = int(y - plant_height // 2)
                
                # Paste the plant onto the garden photograph
                output.paste(plant_img, (paste_x, paste_y), plant_img)

                # We extract the alpha channel of the plant (the actual non-transparent pixels)
                plant_alpha = plant_img.split()[3]
                
                # We paste that alpha channel as WHITE onto our BLACK mask
                mask.paste(255, (paste_x, paste_y), plant_alpha)
            except Exception as e:
                logger.warning(f"Failed to place plant {i+1}: {e}")
                continue
        
        # Optionally expand the mask slightly so the AI can draw a shadow AROUND the plant
        mask = mask.filter(ImageFilter.MaxFilter(5))

        out_rgb = output.convert('RGB')
        buf_img = io.BytesIO()
        out_rgb.save(buf_img, format='JPEG', quality=95)
        
        # Save mask as standard Grayscale Png
        buf_mask = io.BytesIO()
        mask.save(buf_mask, format='PNG')
        
        return (
            base64.b64encode(buf_img.getvalue()).decode('utf-8'),
            base64.b64encode(buf_mask.getvalue()).decode('utf-8')
        )
    
    def _generate_blend_prompt(self, plants: List[Dict[str, Any]]) -> str:
        plant_descriptions = [p.get('name', 'a plant') for p in plants]
        plants_text = ", ".join(plant_descriptions) if plant_descriptions else "landscaping plants"
        
        return (
            f"Realistic photograph of {plants_text}. Unedited snapshot, normal daytime backyard lighting. "
            f"Matching shadows."
        )
    
    def _blend_with_stability_inpainting(self, composite_image: str, mask_image: str, prompt: str) -> Dict[str, Any]:
        """Send the image and the mask to Stability AI's INPAINTING endpoint to lock the background."""
        
        response = requests.post(
            f"{self.api_host}/v2beta/stable-image/edit/inpaint",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            files={
                "image": base64.b64decode(composite_image),
                "mask": base64.b64decode(mask_image)
            },
            data={
                "prompt": prompt,
                "output_format": "jpeg",
                "negative_prompt": "anime, cartoon, drawing, painting, 3d render, artificial",
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Stability API failed: {response.text}")
            raise Exception(f"Stability API Error {response.status_code}: {response.text}")
            
        data = response.json()
        return {
            'image_base64': data.get("image"),
            'revised_prompt': prompt,
        }

    def quick_blend(self, garden_image: str, plant_image: str, x: float = 0.5, y: float = 0.5, scale: float = 1.0) -> Dict[str, Any]:
        plants = [{'image': plant_image, 'x': x, 'y': y, 'scale': scale}]
        return self.compose_plants(garden_image, plants)
