import json
import logging
import os
import io
import base64
from typing import List, Dict, Any, Tuple
from urllib.request import urlopen
from urllib.parse import urlparse

# Image processing
from PIL import Image

# OpenAI for DALL-E
from openai import OpenAI

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PlantComposer:
    """
    AI-powered plant image composer that blends multiple plants into a garden image.
    
    Usage:
        1. User places plants on garden image in frontend (sends coordinates)
        2. Backend composites plants at specified positions
        3. DALL-E blends everything for natural look
        4. Returns final blended image
    """
    
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        # Using dall-e-3 for best quality blending
        self.model = 'dall-e-3'
        # dall-e-2 is faster and cheaper but lower quality
        self.fast_model = 'dall-e-2'
    
    def compose_plants(
        self,
        garden_image: str,  # URL or base64 or file path
        plants: List[Dict[str, Any]],
        # plants format: [{"image": "url/base64/path", "x": 0.5, "y": 0.5, "scale": 1.0}, ...]
        # x, y are relative coordinates (0-1) from top-left
        # scale is optional, defaults to 1.0
        quality: str = 'standard',  # 'standard', 'high', 'low'
        size: str = '1024x1024',    # '1024x1024', '1792x1024', '1024x1792'
        blend_prompt: str = None,
        return_combined_only: bool = False
    ) -> Dict[str, Any]:
        """
        Main method to compose plants into garden and blend them.
        
        Args:
            garden_image: URL, base64, or file path of the garden/base image
            plants: List of plant dicts with image and coordinates
            quality: 'standard', 'high', or 'low' (affects cost)
            size: Output image size
            blend_prompt: Custom prompt for DALL-E blending (optional)
            return_combined_only: If True, returns just the composited image before AI blending
            
        Returns:
            Dict with:
                - 'composite_image': base64 of the initial composite
                - 'blended_image': base64 of the final AI-blended image
                - 'revised_prompt': The prompt DALL-E used
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
        
        # Step 3: Send to DALL-E for natural blending (ONE API CALL)
        logger.info("Sending to DALL-E for AI blending...")
        blended_result = self._blend_with_dalle(
            composite_image=composite_base64,
            prompt=blend_prompt,
            quality=quality,
            size=size
        )
        
        return {
            'composite_image': composite_base64,
            'blended_image': blended_result['image_base64'],
            'revised_prompt': blended_result.get('revised_prompt', blend_prompt)
        }
    
    def _load_image(self, image_source: str) -> Image.Image:
        """Load an image from URL, base64, or file path."""
        try:
            if image_source.startswith('data:image'):
                # Base64 image
                header, b64data = image_source.split(',', 1)
                img_data = base64.b64decode(b64data)
                return Image.open(io.BytesIO(img_data)).convert('RGBA')
            elif image_source.startswith('http://') or image_source.startswith('https://'):
                # URL image
                with urlopen(image_source) as response:
                    return Image.open(response).convert('RGBA')
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
        # Load garden/base image
        garden = self._load_image(garden_image)
        original_width, original_height = garden.size
        
        logger.info(f"Garden image size: {original_width}x{original_height}")
        
        # Create output image (RGBA for transparency support)
        output = Image.new('RGBA', (original_width, original_height), (0, 0, 0, 0))
        output.paste(garden, (0, 0))
        
        # Place each plant at its position
        for i, plant in enumerate(plants):
            try:
                plant_img = self._load_image(plant['image'])
                
                # Get position (relative 0-1 or absolute)
                x = plant.get('x', 0.5)
                y = plant.get('y', 0.5)
                
                # Convert relative to absolute if needed
                if x <= 1 and y <= 1:
                    x = int(x * original_width)
                    y = int(y * original_height)
                
                # Get scale
                scale = plant.get('scale', 1.0)
                if scale != 1.0:
                    new_width = int(plant_img.width * scale)
                    new_height = int(plant_img.height * scale)
                    plant_img = plant_img.resize((new_width, new_height), Image.LANCZOS)
                
                # Calculate position (center the plant on the point)
                plant_width, plant_height = plant_img.size
                paste_x = x - plant_width // 2
                paste_y = y - plant_height // 2
                
                # Paste plant (using alpha channel as mask)
                output.paste(plant_img, (paste_x, paste_y), plant_img)
                
                logger.info(f"Planted plant {i+1} at ({paste_x}, {paste_y})")
                
            except Exception as e:
                logger.warning(f"Failed to place plant {i+1}: {e}")
                continue
        
        # Convert to RGB for DALL-E (DALL-E doesn't support RGBA)
        output_rgb = output.convert('RGB')
        
        # Save to bytes as base64
        buffer = io.BytesIO()
        output_rgb.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def _generate_blend_prompt(self, plants: List[Dict[str, Any]]) -> str:
        """Generate a prompt for DALL-E to blend plants naturally."""
        plant_descriptions = []
        
        for plant in plants:
            desc = plant.get('description', 'a plant')
            if plant.get('name'):
                desc = f"{plant['name']} - {desc}"
            plant_descriptions.append(desc)
        
        plants_text = ", ".join(plant_descriptions) if plant_descriptions else "plants"
        
        prompt = (
            f"Photo of a garden with {plants_text} placed naturally. "
            f"The plants are seamlessly blended into the garden scene with proper lighting, "
            f"shadows, and perspective matching the original garden image. "
            f"Realistic, photorealistic quality, natural colors, proper depth of field."
        )
        
        return prompt
    
    def _blend_with_dalle(
        self,
        composite_image: str,
        prompt: str,
        quality: str = 'standard',
        size: str = '1024x1024'
    ) -> Dict[str, Any]:
        """
        Send composite image to DALL-E for AI blending.
        Uses DALL-E 3's image editing capability.
        """
        try:
            # DALL-E 3 image edit
            response = self.client.images.generate(
                model=self.model,
                prompt=prompt,
                image=io.BytesIO(base64.b64decode(composite_image)),
                size=size,
                quality=quality,
                n=1
            )
            
            result = response.data[0]
            
            # Get the revised prompt if DALL-E modified it
            revised_prompt = getattr(result, 'revised_prompt', prompt)
            
            # Download and convert result to base64
            image_url = result.url
            with urlopen(image_url) as response:
                image_data = response.read()
            
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            logger.info(f"DALL-E blending complete, revised prompt: {revised_prompt[:100]}...")
            
            return {
                'image_base64': image_base64,
                'revised_prompt': revised_prompt,
                'url': image_url
            }
            
        except Exception as e:
            logger.error(f"DALL-E blending failed: {e}")
            raise
    
    def quick_blend(
        self,
        garden_image: str,
        plant_image: str,
        x: float = 0.5,
        y: float = 0.5,
        scale: float = 1.0
    ) -> Dict[str, Any]:
        """
        Quick method for single plant - convenience wrapper.
        """
        plants = [{
            'image': plant_image,
            'x': x,
            'y': y,
            'scale': scale
        }]
        
        return self.compose_plants(garden_image, plants)


# Standalone testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment")
        exit(1)
    
    print("Initializing PlantComposer for testing...")
    composer = PlantComposer(openai_api_key=api_key)
    
    # Example usage with local files or URLs
    # Replace with actual image paths/URLs
    print("\nUsage:")
    print("composer.compose_plants(")
    print("    garden_image='path/to/garden.jpg',")
    print("    plants=[")
    print("        {'image': 'path/to/plant1.png', 'x': 0.3, 'y': 0.7, 'scale': 0.8},")
    print("        {'image': 'path/to/plant2.png', 'x': 0.6, 'y': 0.5, 'scale': 1.0}")
    print("    ]")
    print(")")
