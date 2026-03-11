I've created a new Django app called "composer" for blending plant images into garden images using just your OpenAI API key.

**Created files:**
- [`composer/__init__.py`](composer/__init__.py) - App init
- [`composer/apps.py`](composer/apps.py) - Django app config
- [`composer/engine.py`](composer/engine.py) - Main AI blending engine
- [`composer/management/commands/run_composer.py`](composer/management/commands/run_composer.py) - CLI command

**How it works (ONE API call):**

1. Frontend sends: garden image + list of plants with coordinates
2. Backend composites plants at specified x,y positions
3. DALL-E 3 blends everything naturally in ONE API call
4. Returns the final blended image

**Frontend API call example:**
```python
composer.compose_plants(
    garden_image='https://example.com/garden.jpg',
    plants=[
        {'image': 'plant1.png', 'x': 0.3, 'y': 0.7, 'scale': 0.8},
        {'image': 'plant2.png', 'x': 0.6, 'y': 0.5, 'scale': 1.0}
    ]
)
```

**x, y are relative coordinates (0-1)** from top-left of the garden image.

**To use from Django views:**
```python
from ai_plant_engine.composer.engine import PlantComposer

composer = PlantComposer(openai_api_key=os.environ['OPENAI_API_KEY'])
result = composer.compose_plants(garden_image=garden_url, plants=plant_list)
# result['blended_image'] contains base64 of final image
```

**You only need:** OpenAI API key (already have it)