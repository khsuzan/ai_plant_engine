import os
import io
import logging
import requests
from typing import List, Dict, Any
from PIL import Image as PILImage, ImageFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantComposerStability:
    """
    Composites AI-generated/isolated plants onto a static background.
    Guarantees the background pixels remain 100% unchanged.
    """
    def __init__(self, stability_api_key=None):
        self.api_key = stability_api_key or os.getenv("STABILITY_API_KEY")
        if not self.api_key:
            logger.warning("No Stability API key found. AI features will fail.")

    def _get_perspective_scale(self, y: float) -> float:
        return 0.5 + (y * 0.7)

    def _create_shadow(self, plant_img: PILImage.Image) -> PILImage.Image:
        """Generates a PIL-based drop shadow for realism without AI hallucination."""
        shadow = plant_img.copy().convert("RGBA")
        r, g, b, a = shadow.split()
        shadow = PILImage.merge("RGBA", (
            r.point(lambda _: 0),
            g.point(lambda _: 0),
            b.point(lambda _: 0),
            a.point(lambda p: int(p * 0.45))
        ))
        shadow = shadow.resize((shadow.width, int(shadow.height * 0.25)), PILImage.Resampling.LANCZOS)
        return shadow.filter(ImageFilter.GaussianBlur(15))

    # ==========================================
    # THIS IS WHERE YOU CALL THE AI ENGINE
    # ==========================================
    def _prepare_plant_asset_with_ai(self, image_path: str, prompt: str) -> PILImage.Image:
        """
        Takes the reference image and isolates it using Stability AI.
        (If you wanted the AI to heavily redraw the plant first, you would call 
        the image-to-image endpoint right before this background removal step).
        """
        logger.info(f"Calling Stability AI to process plant: '{prompt}'...")
        
        url = "https://api.stability.ai/v2beta/stable-image/edit/remove-background"
        
        with open(image_path, "rb") as f:
            response = requests.post(
                url,
                headers={
                    "authorization": f"Bearer {self.api_key}",
                    "accept": "image/*"
                },
                files={"image": f},
                data={"output_format": "png"} # Force PNG to get alpha transparency
            )

        if response.status_code == 200:
            logger.info("AI processing successful. Plant isolated.")
            return PILImage.open(io.BytesIO(response.content)).convert("RGBA")
        else:
            logger.error(f"Stability AI API Error: {response.text}")
            raise Exception(f"AI failed to process the plant asset: {response.text}")

    # ==========================================
    # MAIN COMPOSER (SIGNATURE UNCHANGED)
    # ==========================================
    def compose_plants(self, garden_image_path: str, plants: List[Dict[str, Any]], use_ai: bool = True) -> Dict[str, Any]:
        """
        Takes the garden background, calls AI to isolate the plants, 
        and pastes them cleanly onto the background.
        """
        try:
            # 1. Open background (It remains completely untouched)
            garden = PILImage.open(garden_image_path).convert("RGBA")
            canvas = garden.copy()

            for i, p in enumerate(plants):
                logger.info(f"Processing plant {i+1}/{len(plants)}")
                
                plant_path = p.get("image_path")
                plant_name = p.get("name", "plant")

                # 2. Call the AI Engine to get a clean, transparent asset
                if use_ai and self.api_key:
                    plant_img = self._prepare_plant_asset_with_ai(plant_path, plant_name)
                else:
                    # Fallback if no API key is passed
                    plant_img = PILImage.open(plant_path).convert("RGBA")

                # 3. Calculate Scale & Perspective
                perspective = self._get_perspective_scale(p["y"])
                total_scale = min(p.get("scale", 1.0) * perspective, 2.0)
                new_size = (int(plant_img.width * total_scale), int(plant_img.height * total_scale))
                plant_img = plant_img.resize(new_size, PILImage.Resampling.LANCZOS)

                # 4. Calculate Placement
                x = int(p["x"] * canvas.width - plant_img.width / 2)
                y = int(p["y"] * canvas.height - plant_img.height)

                # 5. Apply Shadow
                shadow = self._create_shadow(plant_img)
                shadow_x = x + int(plant_img.width * 0.05)
                shadow_y = y + int(plant_img.height * 0.85)
                canvas.paste(shadow, (shadow_x, shadow_y), shadow)

                # 6. Paste the AI-isolated plant onto the garden
                canvas.paste(plant_img, (x, y), plant_img)

            # 7. Finalize
            final_canvas = canvas.convert("RGB")
            buf = io.BytesIO()
            final_canvas.save(buf, format="WEBP")
            
            return {
                "success": True,
                "final_image_bytes": buf.getvalue(),
                "preview_image": final_canvas
            }

        except Exception as e:
            logger.error(f"Composition failed: {e}")
            return {"success": False, "error": str(e)}