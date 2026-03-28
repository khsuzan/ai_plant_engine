from django.core.management.base import BaseCommand
from ...engine import PlantComposer
import os
import json


class Command(BaseCommand):
    help = 'Run AI Plant Composer to blend plants into garden images'

    def add_arguments(self, parser):
        parser.add_argument(
            '--garden',
            type=str,
            help='Path or URL to garden/base image',
            required=True
        )
        parser.add_argument(
            '--plants',
            type=str,
            help='JSON string of plants with coordinates: [{"image": "url/path", "x": 0.5, "y": 0.5, "scale": 1.0}, ...]',
            required=True
        )
        parser.add_argument(
            '--output',
            type=str,
            help='Output file path for the result image',
            default='output.jpg'
        )
        parser.add_argument(
            '--quality',
            type=str,
            choices=['standard', 'high', 'low'],
            default='standard',
            help='DALL-E quality level (affects cost)'
        )
        parser.add_argument(
            '--size',
            type=str,
            choices=['1024x1024', '1792x1024', '1024x1792'],
            default='1024x1024',
            help='Output image size'
        )
        parser.add_argument(
            '--prompt',
            type=str,
            help='Custom blend prompt (optional)'
        )
        parser.add_argument(
            '--api-key',
            type=str,
            help='OpenAI API key (optional, uses OPENAI_API_KEY env if not provided)'
        )

    def handle(self, *args, **options):
        # Get API key
        api_key = options.get('api_key') or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            self.stdout.write(self.style.ERROR('ERROR: OpenAI API key required. Set --api-key or OPENAI_API_KEY environment variable'))
            return

        # Parse plants JSON
        try:
            plants = json.loads(options['plants'])
        except json.JSONDecodeError as e:
            self.stdout.write(self.style.ERROR(f'ERROR: Invalid JSON for plants: {e}'))
            return

        # Initialize composer
        composer = PlantComposer(openai_api_key=api_key)
        
        self.stdout.write(self.style.SUCCESS(f'Composing {len(plants)} plants into garden...'))

        try:
            result = composer.compose_plants(
                garden_image_path=options['garden'],
                plants=plants,
                size=options['size'],
            )

            if result.get('blended_image'):
                # Save the blended image
                import base64
                image_data = base64.b64decode(result['blended_image'])
                with open(options['output'], 'wb') as f:
                    f.write(image_data)
                
                self.stdout.write(self.style.SUCCESS(f'Successfully saved to: {options["output"]}'))
                self.stdout.write(f'Revised prompt: {result.get("revised_prompt", "N/A")[:100]}...')
            else:
                self.stdout.write(self.style.ERROR('No blended image returned'))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error: {e}'))
