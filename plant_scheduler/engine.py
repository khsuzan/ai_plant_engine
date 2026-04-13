import datetime
import json
import logging
import os
from typing import Any, Dict

from openai import OpenAI

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PlantScheduler:
    """
    AI-powered task scheduler for garden projects.
    Uses OpenAI to generate a comprehensive JSON schedule based on garden and plant data.
    """

    def __init__(self, **kwargs):
        self.api_key = (
            kwargs.get("openai_api_key")
            or kwargs.get("api_key")
            or os.getenv("OPENAI_API_KEY")
        )
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4-turbo"  # Recommended model for JSON and complex reasoning

    def _build_prompt(self, garden_data: Dict[str, Any], start_date: str) -> str:
        """
        Builds the text prompt instructing the AI to create a schedule.
        """
        # Convert garden data to a formatted string
        data_str = json.dumps(garden_data, indent=2)

        prompt = f"""
    Based on the following garden and plant data, create a detailed planting and maintenance schedule. 
    The garden is located in {garden_data.get("location", "an unknown location")}.

    ### IMPORTANT:
    The schedule MUST start from {start_date}. All milestones and tasks should follow this timeline sequentially.

    ### GARDEN DATA:
    {data_str}

### REQUIREMENTS:
1. Create a week-by-week timeline of milestones (e.g., Week 1: Soil Preparation, Week 2: Planting, etc.).
2. For each task, include specific details based on the plant's needs (water, sunlight, spacing, season, care guide).
3. Identify required tools and materials for the tasks.
4. Output MUST be valid JSON matching the following structure exactly:

{{
  "milestones": [
    {{
      "week": 1,
      "title": "Soil Preparation",
      "description": "General description of what happens this week",
      "start_date": "2024-03-10",
      "end_date": "2024-03-16",
      "tasks": [
        {{
          "title": "Task Name",
          "type": "Planting | Watering | Pruning | Fertilizing | Preparation",
          "priority": "High | Medium | Low",
          "status": "Pending",
          "duration_minutes": 30,
          "tools_needed": ["Tool 1", "Tool 2"],
          "materials": ["Material 1"],
          "notes": "Specific instructions based on plant care guide"
        }}
      ]
    }}
  ]
}}

Generate a schedule covering the initial setup and planting, plus the first few weeks of maintenance based on the plants provided.
"""
        return prompt.strip()

    def generate_schedule(
        self, garden_data: Dict[str, Any], start_date: str = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Generates a task schedule based on the provided garden data.
        Returns a parsed JSON dictionary.
        """
        logger.info(
            f"Generating schedule for garden: {garden_data.get('name', 'Unknown')} using {self.model}"
        )
        if not start_date:
            start_date = datetime.date.today().isoformat()

        prompt = self._build_prompt(garden_data, start_date)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert horticulturist and garden planner. Your task is to generate a comprehensive garden planting and maintenance schedule in valid JSON format.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )

            result_content = response.choices[0].message.content

            # Parse the JSON response
            schedule_data = json.loads(result_content)
            return schedule_data

        except Exception as e:
            logger.error(f"Failed to generate schedule: {e}")
            raise
