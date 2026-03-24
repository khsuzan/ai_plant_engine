import json
import os
from django.core.management.base import BaseCommand

from ...engine import PlantScheduler

class Command(BaseCommand):
    help = 'Run the AI plant scheduler with sample garden data'

    def handle(self, *args, **options):
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            self.stderr.write(self.style.ERROR('OPENAI_API_KEY environment variable not set. Please set it to test the scheduler.'))
            return

        scheduler = PlantScheduler(openai_api_key=api_key)
        
        # Sample data based on user prompt
        sample_garden_data = {
            "id": 2,
            "name": "Front Yard Project",
            "location": "Dhaka, Bangladesh",
            "sunlight": "full_sun",
            "soil_type": "loam",
            "garden_type": "flower",
            "plants": [
                {
                    "plant": {
                        "common_name": "Chaste Tree",
                        "plant_type": "deciduous shrub or small tree",
                        "water": "Moderate; allow soil to dry out between waterings.",
                        "spacing": "6 to 10 feet apart for optimal growth.",
                        "care_guide": "Chaste Trees thrive in well-drained soil and full sun.",
                        "season": "Spring to Fall"
                    }
                },
                {
                    "plant": {
                        "common_name": "Japanese Knotweed",
                        "plant_type": "perennial herbaceous plant",
                        "water": "Moderate; prefers moist soil but can tolerate drier conditions once established.",
                        "spacing": "1 to 2 meters apart to allow for spreading.",
                        "care_guide": "Japanese Knotweed requires minimal care once established. It thrives in a variety of soil conditions and prefers moist environments.",
                        "season": "grows actively in spring and summer; dormant in winter."
                    }
                }
            ]
        }

        self.stdout.write(self.style.SUCCESS('Generating plant schedule from AI...'))
        
        try:
            result = scheduler.generate_schedule(sample_garden_data)
            self.stdout.write(self.style.SUCCESS('Successfully generated schedule!'))
            self.stdout.write(json.dumps(result, indent=2))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f'Error generating schedule: {e}'))
