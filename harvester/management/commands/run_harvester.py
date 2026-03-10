import os
import logging
from django.core.management.base import BaseCommand
from django.conf import settings
from ...engine import HarvestEngine

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Runs the 4-Phase AI Recursive Harvester pipeline to seed the Database.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--count',
            type=int,
            default=50,
            help='Number of plants to harvest in this batch.',
        )

    def handle(self, *args, **options):
        target_count = options['count']
        
        # Load your API Key correctly - perhaps from settings or os.environ
        api_key = getattr(settings, 'OPENAI_API_KEY', os.environ.get('OPENAI_API_KEY'))
        
        if not api_key:
            self.stdout.write(self.style.ERROR("API Key not found. Please set OPENAI_API_KEY."))
            return

        self.stdout.write(self.style.NOTICE(f"Starting Recursive Harvester for {target_count} plants constraint..."))

        try:
            engine = HarvestEngine(openai_api_key=api_key)
            engine.harvest_plants(target_count=target_count)
            self.stdout.write(self.style.SUCCESS("Harvesting process finished successfully."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Harvester crashed: {e}"))
            logger.exception("Harvester failed.")
