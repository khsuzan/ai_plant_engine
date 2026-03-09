# Calling the HarvestEngine from a Real Backend

To integrate the AI Engine into a production Django (or any Python) backend, you need to instantiate the `HarvestEngine`, fetch your real database records to avoid duplicates, and call the main pipeline function.

## 1. The Target Function
The sole entry point for your background task or cron job should be:

```python
engine.harvest_plants(target_count=5, existing_plants_list=existing_names)
```

## 2. Example Django Integration (e.g. inside a Celery task, Cron, or Custom Command)

Here is a full example of what it looks like when wired to a real database:

```python
import os
from django.conf import settings
from your_main_app.models import Plant 
from harvester.engine import HarvestEngine

def trigger_daily_harvest():
    # 1. Initialize the Engine
    api_key = os.environ.get("OPENAI_API_KEY", settings.OPENAI_API_KEY)
    engine = HarvestEngine(openai_api_key=api_key)
    
    # 2. Query your REAL base to find all existing plants
    # This ensures the AI model strictly skips generating them
    existing_names = list(Plant.objects.values_list('common_name', flat=True))
    
    # 3. Call the pipeline
    # The pipeline will generate 'target_count' brand new plants 
    # and hit `_phase_4_storage` for each one under the hood
    target_count = 10
    
    try:
        engine.harvest_plants(target_count=target_count, existing_plants_list=existing_names)
        print("Daily harvest successfully fully completed!")
    except Exception as e:
        print(f"Daily harvest failed: {e}")
```

## 3. How Data Enters Your Real Database (`_phase_4_storage`)

When `harvest_plants` executes, it internally calls `_phase_4_storage(structured_data)` for every new plant it finds. **You need to update `harvester/engine.py`** to actively save to your DB.

Find this function in `harvester/engine.py`:
```python
    def _phase_4_storage(self, structured_data: Dict[str, Any]) -> None:
        """
        Phase 4: Save it to the Django Models.
        """
        # UNCOMMENT AND EDIT THIS FOR YOUR REAL BACKEND:
        
        # from your_main_app.models import Plant
        # 
        # Plant.objects.update_or_create(
        #     common_name=structured_data.get('common_name'),
        #     defaults=structured_data
        # )
        
        print(f"Mock Data Saved:\n{json.dumps(structured_data, indent=2)}\n")
```

Once that is un-commented, calling `engine.harvest_plants()` will fully automate generating new plants and saving them instantly to Postgres/SQLite.
