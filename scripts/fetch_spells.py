#fetch.py

import json
import os
from pathlib import Path

# --- Configuration ---
#OPEN5e_DATASET_FILE = "dataset_open5e.json"
#HOMEBREW_FILE = "homebrew_spells.json"
#OUTPUT_FILE = "formatted_dnd_spells.jsonl"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OPEN5e_DATASET_FILE = DATA_DIR / "dataset_open5e.json"
HOMEBREW_FILE = DATA_DIR / "homebrew_spells.json"
OUTPUT_FILE = DATA_DIR / "formatted_dnd_spells.jsonl"

def load_spells_from_file(path: str) -> list:
    """
    Helper function to load a list of spells from a given JSON file.
    This version is updated to handle both list and dictionary formats.
    """
    if not os.path.exists(path):
        print(f"Info: File not found at '{path}'. Skipping.")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check if the loaded data is a dictionary containing a 'spells' key
        if isinstance(data, dict) and 'spells' in data:
            spell_list = data['spells']
            print(f"Loaded {len(spell_list)} spells from '{path}' (from 'spells' key).")
            return spell_list
        # Check if the loaded data is already a list
        elif isinstance(data, list):
            print(f"Loaded {len(data)} spells from '{path}' (from list).")
            return data
        # Handle unknown formats
        else:
            print(f"Warning: Could not find a valid spell list in '{path}'. Skipping.")
            return []

    except json.JSONDecodeError:
        print(f"Error: The file '{path}' is not a valid JSON file. Skipping.")
        return []

def normalize_spell(spell):
    """Converts a spell into a standard dictionary format."""
    return {
        "name": spell.get("name", "Unknown Spell"),
        "level": spell.get("level", "Unknown Level"),
        "school": spell.get("school", "Unknown School"),
        "casting_time": spell.get("casting_time", "1 action"),
        "range": spell.get("range", "Self"),
        "components": spell.get("components", "V, S"),
        "duration": spell.get("duration", "Instantaneous"),
        "dnd_class": spell.get("dnd_class", ", ".join(spell.get("classes", []))),
        "desc": spell.get("desc", spell.get("description", "No description.")),
        "rationale": spell.get("rationale", "")
    }

def convert_to_chat_format(spells):
    """Formats the dataset into the Llama 3 chat template."""
    print("Converting spells to Llama 3 chat format...")
    formatted_data = []
    for spell in spells:
        s = normalize_spell(spell)
        instruction = (
                "Generate a balanced D&D 5e spell based on the following details.\n"
                f"Level: {s['level']}\n"
                f"School: {s['school']}\n"
                "Ensure the mechanics are thematically consistent with the spell's school and level."
        )
        
        completion = (
                f"Name: {s['name']}\n"
                f"Level: {s['level']}\n"
                f"School: {s['school']}\n"
                f"Casting Time: {s['casting_time']}\n"
                f"Range: {s['range']}\n"
                f"Components: {s['components']}\n"
                f"Duration: {s['duration']}\n"
                f"Classes: {s['dnd_class']}\n"
                f"Description: {s['desc'].strip()}"
        )

        if s["rationale"]:
            completion += f"\n\nRationale: {s['rationale']}"

        formatted_text = (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{completion}<|eot_id|>"
        )
        formatted_data.append({"text": formatted_text})

    return formatted_data

def save_to_jsonl(data, path=OUTPUT_FILE):
    """Saves the formatted data to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f" Saved {len(data)} formatted entries to '{path}'")


if __name__ == "__main__":
    # Load spells from both sources using the updated, robust function.
    golden_spells = load_spells_from_file(OPEN5e_DATASET_FILE)
    homebrew_spells = load_spells_from_file(HOMEBREW_FILE)

    # This line will now work correctly as both variables will be lists.
    combined_spells = golden_spells + homebrew_spells

    # De-duplicate the list, preferring the homebrew version if names conflict.
    unique_spells = {}
    for spell in combined_spells:
        if 'name' in spell and spell['name']:
            unique_spells[spell['name']] = spell

    final_dataset = list(unique_spells.values())

    print(f"✨ Combined and de-duplicated dataset contains {len(final_dataset)} unique spells.")

    # Proceed with formatting and saving as before.
    if final_dataset:
        formatted_dataset = convert_to_chat_format(final_dataset)
        save_to_jsonl(formatted_dataset)
    else:
        print("No spells were loaded. Please ensure 'golden_dataset.json' exists.")
