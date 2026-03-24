# Plant Scheduler

I've created a new Django app called `plant_scheduler` that generates a detailed planting and maintenance schedule based on a garden and its plants using the OpenAI API.

**Created files:**
- [`plant_scheduler/__init__.py`](plant_scheduler/__init__.py) - App init
- [`plant_scheduler/apps.py`](plant_scheduler/apps.py) - Django app config
- [`plant_scheduler/engine.py`](plant_scheduler/engine.py) - Main AI scheduling engine
- [`plant_scheduler/management/commands/run_scheduler.py`](plant_scheduler/management/commands/run_scheduler.py) - CLI test command

**How it works:**

1. The frontend or backend provides the complete garden JSON data (including location, soil type, and an array of plants).
2. The `PlantScheduler` engine parses the garden data and constructs a comprehensive system prompt for the AI to schedule the plant tasks.
3. OpenAI (`gpt-4-turbo`) generates a structured JSON schedule containing weekly milestones, tasks, tools needed, materials, due dates, and durations.
4. Returns the JSON schedule.

**To use from Django views:**
```python
from plant_scheduler.engine import PlantScheduler
import os

# Initialize the scheduler
scheduler = PlantScheduler(openai_api_key=os.environ['OPENAI_API_KEY'])

# Provide garden data
garden_data = {
    "name": "Front Yard Project",
    "location": "Dhaka, Bangladesh",
    "plants": [ ... ]
}

# Generate schedule
schedule_json = scheduler.generate_schedule(garden_data)

# `schedule_json` contains the parsed dictionary of milestones and tasks.
```

**You only need:**
- OpenAI API key (ensure it is exported to your environment as `OPENAI_API_KEY`).
