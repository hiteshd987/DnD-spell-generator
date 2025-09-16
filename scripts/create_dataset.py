# create_dataset.py

from pathlib import Path
import json
import os
import requests
import time
from openai import OpenAI
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv(dotenv_path="../.env")

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        raise ValueError("OpenAI API key not found.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please make sure your .env file exists and contains a valid OPENAI_API_KEY.")
    exit()

# Configuration constants
#OUTPUT_FILE = "golden_dataset.json"
#OUTPUT_FILE = os.path.join("data", "dataset_open5e.json")
# make sure the parent folder exist
#os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "dataset_open5e.json"

#API_MODEL = "gpt-4o-mini"
API_MODEL = "gpt-4.1"
REQUEST_DELAY = 5
MAX_SPELLS_TO_PROCESS = 5


def fetch_srd_spells() -> list:
    """
    Fetches all available spells from the Open5e API.
    The base endpoint now defaults to providing the SRD spell list.
    """
    print("Fetching all available SRD spells from the main API endpoint...")

    # Used the base URL without any source filters.
    url = "https://api.open5e.com/v1/spells/?limit=1000"

    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        all_spells = data.get("results", [])
        print(f"Found {len(all_spells)} spells.")
        return all_spells
    except requests.exceptions.RequestException as e:
        print(f"Error fetching spells from API: {e}")
        return []


def generate_rationale_for_spell(spell: dict) -> str | None:
    """Uses an LLM to generate a design rationale for a given spell (unchanged)."""

    spell_summary = (
        f"Name: {spell.get('name')}\n"
        f"Level: {spell.get('level')}\n"
        f"School: {spell.get('school')}\n"
        f"Casting Time: {spell.get('casting_time')}\n"
        f"Range: {spell.get('range')}\n"
        f"Duration: {spell.get('duration')}\n"
        f"Description: {spell.get('desc')}"
    )

    prompt = (
        "You are an expert D&D 5e game designer. Your task is to analyze the following spell "
        "and write a 'Rationale' section explaining its design choices. Be concise but specific.\n\n"
        "Cover the following points in your rationale:\n"
        "1.  **School Justification:** Why does it belong to this school of magic?\n"
        "2.  **Level Justification:** How does its power level fit with other spells of the same level?\n"
        "3.  **Key Mechanics:** Explain why a specific save, duration, or component is used.\n"
        "4.  **Balancing Factor:** What is the key limitation (e.g., range, concentration, cost) that balances the spell?\n\n"
        f"--- SPELL DATA ---\n{spell_summary}\n\n"
        "--- RATIONALE ---\n"
    )

    try:
        print(f"   > Sending request to {API_MODEL} for rationale...")
        response = client.chat.completions.create(
            model=API_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert D&D 5e game designer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=250,
        )
        rationale = response.choices[0].message.content
        print("   > Rationale received.")
        return rationale.strip()
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return None


def main():
    """Main function to orchestrate the dataset creation process."""
    print("--- Dataset Creation  ---")
    print(f"This script will process up to {MAX_SPELLS_TO_PROCESS} spells using the '{API_MODEL}' model.")
    print("This will make API calls to OpenAI, which may incur costs. \n")

    golden_spells = []
    processed_spell_names = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            golden_spells = json.load(f)
            processed_spell_names = {spell['name'] for spell in golden_spells}
        print(f"Loaded {len(golden_spells)} spells from existing '{OUTPUT_FILE}'.\n")  

    # Call the updated fetching function
    candidate_spells = fetch_srd_spells()

    spells_to_process = [
        spell for spell in candidate_spells
        if spell['name'] not in processed_spell_names
    ]

    if not spells_to_process:
        if not candidate_spells:
            print("Could not fetch any spells. Please check your internet connection and the API status.")
        else:
            print("All available spells are already processed. Nothing more to do.")
        return

    processed_count = 0
    for spell in spells_to_process:
        if len(golden_spells) >= MAX_SPELLS_TO_PROCESS:
            print(f"\nReached the limit of {MAX_SPELLS_TO_PROCESS} spells. Halting.")
            break

        print(f"\nProcessing spell ({len(golden_spells) + 1}/{MAX_SPELLS_TO_PROCESS}): {spell['name']}...")

        rationale = generate_rationale_for_spell(spell)

        if rationale:
            spell_with_rationale = spell.copy()
            spell_with_rationale['rationale'] = rationale
            golden_spells.append(spell_with_rationale)
            processed_count += 1

            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(golden_spells, f, indent=2)
            print(f"   > Saved to '{OUTPUT_FILE}'.")

        print(f"   > Waiting for {REQUEST_DELAY} seconds...")
        time.sleep(REQUEST_DELAY)

    print("\n--- ✨ Process Complete! ---")
    print(f"Successfully added rationales for {processed_count} new spells.")
    print(f"Your golden dataset now contains {len(golden_spells)} total spells.")
    print(f"The file '{OUTPUT_FILE}' is ready to be used for fine-tuning.")

if __name__ == "__main__":
    main()
