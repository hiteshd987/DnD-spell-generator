#generate.py

import torch
from pathlib import Path
from transformers import pipeline
import re
import os

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

def parse_concept(text: str) -> dict | None:
    """Parses the generated concept text to extract level, school, and theme."""
    concept = {}
    try:
        for line in text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                if key in ['level', 'school', 'theme']:
                    concept[key] = value

        if all(key in concept for key in ['level', 'school', 'theme']):
            return concept
    except Exception:
        pass

    print(" Warning: Could not parse the generated concept. Trying again.")
    return None

def clean_final_output(text: str) -> str:
    """Cleans the generated spell text."""
    text = re.split(r'\n\nRationale:', text, flags=re.IGNORECASE)[0]
    text = re.split(r'\n\nThis is a complete D&D 5e spell', text, flags=re.IGNORECASE)[0]
    return text.strip()

def generate_creative_spell(model_pipeline):
    """Generates a new, unique spell in a two-step process."""
    print("\nStep 1: Coming up with a unique spell concept...")

    # Step 1: Generate the Concept
    concept_prompt = (
        "Generate a creative and unique concept for a new D&D 5e spell. "
        "Provide only its Level, School, and a brief Theme in the format:\n"
        "Level: <level>\n"
        "School: <school>\n"
        "Theme: <theme>"
    )

    concept_full_prompt = model_pipeline.tokenizer.apply_chat_template(
        [{"role": "user", "content": concept_prompt}],
        tokenize=False,
        add_generation_prompt=True
    )

    terminators = [
        model_pipeline.tokenizer.eos_token_id,
        model_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    concept_output = model_pipeline(
        concept_full_prompt,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        eos_token_id=terminators,
        pad_token_id=model_pipeline.tokenizer.eos_token_id
    )[0]['generated_text']

    try:
        concept_text = concept_output.split("<|start_header_id|>assistant<|end_header_id|>")[1].strip()
        concept = parse_concept(concept_text)
        if not concept:
            return
    except IndexError:
        print(" Warning: Failed to generate a valid concept structure. Trying again.")
        return

    print(f"   > Concept received: {concept['level']} {concept['school']} - {concept['theme']}")
    # --- Corrected Print Statement ---
    print("Step 2: Generating the full spell based on the concept...")

    # Step 2: Generate the Full Spell
    spell_instruction = (
        "Generate a complete, balanced D&D 5e spell based on the following details.\n"
        f"Name: {concept.get('name', 'Generate a fitting unique name')}\n"
        f"Level: {concept['level']}\n"
        f"School: {concept['school']}\n"
        f"Classes: {', '.join(concept.get('classes', []))}\n"
        f"Theme: {concept['theme']}\n"
        "Ensure the mechanics and description are thematically consistent. "
        "Do not include a 'Rationale' section in your output."
    )

    spell_full_prompt = model_pipeline.tokenizer.apply_chat_template(
        [{"role": "user", "content": spell_instruction}],
        tokenize=False,
        add_generation_prompt=True
    )

    spell_output = model_pipeline(
        spell_full_prompt,
        max_new_tokens=400,
        temperature=0.75,
        top_p=0.9,
        do_sample=True,
        eos_token_id=terminators,
        pad_token_id=model_pipeline.tokenizer.eos_token_id
    )[0]['generated_text']

    try:
        assistant_response = spell_output.split("<|start_header_id|>assistant<|end_header_id|>")[1].strip()
        cleaned_spell = clean_final_output(assistant_response)
    except IndexError:
        cleaned_spell = "Failed to generate a clean spell output."

    print("\n" + "="*20 + " Generated Spell " + "="*20)
    print(cleaned_spell)
    print("="*57 + "\n")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MODEL_DIR = PROJECT_ROOT / "models" / "dnd-lora-final"
    #model_dir = os.path.join("models","dnd-lora-final")
    print("--- <200d>♂️ D&D Spell Generator (Creative Mode) ---")
    print(f"Loading fine-tuned model from {MODEL_DIR}...")

    try:
        pipe = pipeline(
            "text-generation",
            model=MODEL_DIR,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ) 
    except Exception as e:
        print(f"\n Error loading the model: {e}")
        exit()

    print(" Model loaded successfully.\n")

    while True:
        generate_creative_spell(pipe)

        # --- Corrected Input Statement ---
        another = input("Generate another spell? (y/n): ")
        if another.lower() != 'y':
            break

