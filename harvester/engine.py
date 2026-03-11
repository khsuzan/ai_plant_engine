import json
import logging
import os
from typing import List, Dict, Any

# Using OpenAI for the LLM client
import urllib.request
import urllib.parse

from openai import OpenAI

# Setup basic logging for local testing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HarvestEngine:
    def __init__(self, openai_api_key: str):
        # Configure your AI and Search clients here
        self.client = OpenAI(api_key=openai_api_key)
        # Using gpt-4o-mini is ~10x faster and 10x cheaper than gpt-4o for bulk structural tasks
        self.model = 'gpt-4o-mini' 

    def harvest_plants(self, target_count: int = 5, existing_plants_list: List[str] = None) -> List[Dict[str, Any]]:
        """
        Execute the Direct AI Harvester Pipeline for plants.
        Returns a list of structured data for the generated plants so that insertion can be handled by the caller.
        """
        if existing_plants_list is None:
            existing_plants_list = []
            
        logger.info(f"Starting Harvester Pipeline for {target_count} plants.")
        
        # Phase 1: Ask AI for a list of names we don't have
        plant_names = self._generate_plant_names(target_count, existing_plants_list)
        logger.info(f"Generated {len(plant_names)} raw plant names to harvest.")

        harvested_plants = []
        # Optionally, you can implement concurrent.futures right here to blast these out simultaneously
        for name in plant_names:
            try:
                # Phase 2 & 3: Structure the single plant
                structured_data = self._generate_single_plant(name)
                
                # We skip direct storage here to let the caller handle insertion
                # self._phase_4_storage(structured_data)
                harvested_plants.append(structured_data)
                
                logger.info(f"Successfully processed: {structured_data.get('common_name', name)}")
            except Exception as e:
                logger.error(f"Failed processing plant '{name}': {e}")
                
        return harvested_plants

    def start_background_harvesting(
        self, 
        days: int = 7, 
        plants_per_hour: int = 5, 
        insert_callback=None, 
        get_existing_plants_callback=None
    ):
        """
        Runs a background task to harvest plants periodically (every hour) for a specified number of days.
        """
        import threading
        import time
        from datetime import datetime, timedelta

        def worker():
            end_time = datetime.now() + timedelta(days=days)
            logger.info(f"Starting background harvesting for {days} days ({plants_per_hour} plants/hour). Ends at {end_time}")
            
            while datetime.now() < end_time:
                try:
                    existing_plants = []
                    if get_existing_plants_callback:
                        existing_plants = get_existing_plants_callback()
                        
                    harvested = self.harvest_plants(target_count=plants_per_hour, existing_plants_list=existing_plants)
                    if harvested and insert_callback:
                        insert_callback(harvested)
                except Exception as e:
                    logger.error(f"Error during background harvest task: {e}")
                
                # Check if we have surpassed the end time before sleeping another hour
                if datetime.now() >= end_time:
                    break
                    
                # Sleep for 1 hour (3600 seconds)
                time.sleep(3600)
            logger.info("Background harvesting completed.")
            
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread
    def _get_schema(self) -> str:
        return '''
        {
            "common_name": "string",
            "scientific_name": "string",
            "plant_type": "string",
            "description": "string",
            "main_image_url": "string (URL or empty)",
            "sunlight": "string",
            "water": "string",
            "spacing": "string",
            "growth_size": "string",
            "season": "string",
            "difficulty": "string",
            "care_guide": "string",
            "bloom_spring": "boolean",
            "bloom_summer": "boolean",
            "bloom_fall": "boolean",
            "bloom_winter": "boolean",
            "shopping_link": "string (URL or empty)",
            "tags": "string (comma-separated)",
            "family": "string",
            "propagation": "string"
        }
        '''

    def _generate_plant_names(self, target_count: int, existing_names: List[str]) -> List[str]:
        """
        Ask AI to generate a raw list of plant names, avoiding existing ones via the existing_names prompt injection.
        """
        # If the list of existing names is huge, we don't want to blow up the token count, 
        # so we truncate string formatting if it exceeds typical limits, though a standard DB of names is usually fine.
        existing_names_str = ", ".join(existing_names) if existing_names else "None"

        prompt = (
            f"Generate a list of {target_count} unique plant, flower, or tree common names. "
            f"Do NOT include any of the following plants you have already generated: [{existing_names_str}]. "
            "Return ONLY a valid JSON array of strings (the new plant names). Do not include markdown formatting or any other text."
        )

        try:
            print(f"Asking AI for {target_count} new plant names...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8 # Slightly higher temp for better variety discovery
            )
            cleaned_text = response.choices[0].message.content.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned_text)
        except Exception as e:
            logger.error(f"Failed to generate plants list from AI: {e}")
            return []

    def _fetch_wikipedia_image(self, plant_name: str) -> str:
        """
        Fetches the primary image URL for a given plant from the free Wikipedia API.
        Includes a fallback to Wikipedia's open search if the exact term + redirect fails.
        """
        def fetch_by_title(title: str) -> str:
            try:
                query = urllib.parse.quote(title)
                url = f"https://en.wikipedia.org/w/api.php?action=query&prop=pageimages&titles={query}&redirects=1&pithumbsize=800&format=json"
                req = urllib.request.Request(url, headers={'User-Agent': 'PlantHarvesterApp/1.0'})
                with urllib.request.urlopen(req) as response:
                    data = json.loads(response.read().decode())
                    pages = data.get("query", {}).get("pages", {})
                    for _, page_info in pages.items():
                        if "thumbnail" in page_info:
                            return page_info["thumbnail"]["source"]
            except Exception as e:
                logger.warning(f"Error fetching Wikipedia image for title '{title}': {e}")
            return ""

        # 1. Try exact match (with redirects)
        image_url = fetch_by_title(plant_name)
        if image_url:
            return image_url

        # 2. Fallback: Search for the closest Wikipedia article title
        try:
            search_query = urllib.parse.quote(plant_name)
            search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={search_query}&limit=3&format=json"
            req = urllib.request.Request(search_url, headers={'User-Agent': 'PlantHarvesterApp/1.0'})
            with urllib.request.urlopen(req) as response:
                search_data = json.loads(response.read().decode())
                # OpenSearch format: [Query, [Title 1, Title 2...], [Snippet...], [Link...]]
                if len(search_data) > 1 and len(search_data[1]) > 0:
                    for possible_title in search_data[1]:
                        # Try the first few search results, return first image found
                        if possible_title.strip():
                            fallback_image = fetch_by_title(possible_title)
                            if fallback_image:
                                return fallback_image
        except Exception as e:
            logger.warning(f"Error during fallback search for '{plant_name}': {e}")

        return ""

    def _generate_single_plant(self, plant_name: str) -> Dict[str, Any]:
        """
        Generate data structure for a specific known plant using AI knowledge.
        We prompt the model to act as a deep knowledge botanical expert and fetch real
        Wikipedia images to supplement the structured object.
        """
        prompt = (
            f"You are an expert botanist and web-scraper. I need you to provide the most exact, "
            f"detailed botanical and care information for the plant '{plant_name}'.\n\n"
            f"Ensure scientific names, toxicity, and family domains are flawlessly accurate "
            f"as if extracted directly from Wikipedia or a high-end nursery.\n\n"
        )
        
        real_image = self._fetch_wikipedia_image(plant_name)
        if real_image:
            prompt += f"I have found a verified image for this plant. You MUST use this exact URL for the 'main_image_url' field: {real_image}\n\n"
        else:
            prompt += "If you cannot find a real image URL, leave 'main_image_url' as an empty string. Do NOT invent or use example.com URLs.\n\n"
        
        prompt += (
            f"Format the output strictly as a JSON object matching this schema:\n"
            f"{self._get_schema()}\n\n"
            "Return ONLY valid JSON. Do not include markdown formatting."
        )

        try:
            print(f"[{plant_name}] Structuring data with AI...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            cleaned_text = response.choices[0].message.content.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned_text)
        except Exception as e:
            logger.error(f"Failed to parse structured JSON for {plant_name}: {e}")
            raise e

    def _phase_4_storage(self, structured_data: Dict[str, Any]) -> None:
        """
        Phase 4: Save it to the Django Models.
        """
        # TODO: Import your Django model dynamically to avoid circular issues, or use relative imports
        # from your_main_app.models import Plant
        
        # logger.info(f"Mock Save to DB: {structured_data.get('common_name', 'Unknown')}")
        print(f"Mock Data to Save:\n{json.dumps(structured_data, indent=2)}\n")

if __name__ == "__main__":
    from dotenv import load_dotenv
    
    # Load environment variables from .env file (useful for standalone testing)
    load_dotenv()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not found. Please set it in your .env file.")
        exit(1)
        
    print("Initializing HarvesterEngine for standalone testing...")
    engine = HarvestEngine(openai_api_key=api_key)
    
    import sys
    # If the user provides an argument, treat it as a specific plant name to parse instead of generating a random list
    if len(sys.argv) > 1:
        plant_name = " ".join(sys.argv[1:])
        print(f"\n--- Testing Single Plant: {plant_name} ---")
        try:
            structured_data = engine._generate_single_plant(plant_name)
            engine._phase_4_storage(structured_data)
        except Exception as e:
            print(f"Error testing single plant: {e}")
    else:
        # Running a small pipeline test with just 2 random plants,
        # simulating that we already have snake plant and ZZ plant in the DB so it shouldn't generate them again
        mock_existing_db = ["Snake Plant", "ZZ Plant", "Monstera deliciosa", "Pothos"]
        print(f"\n--- Testing Pipeline Bulk Generator (Target: 2) ---")
        print(f"Mocking DB Database ignores for: {mock_existing_db}\n")
        harvested = engine.harvest_plants(target_count=2, existing_plants_list=mock_existing_db)
        if harvested:
            engine._phase_4_storage(harvested[0])
            print(f"Final Output preview:\n{json.dumps(harvested, indent=2)}")
