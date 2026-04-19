# 🪄 D&D Spell Fine-Tuning (LoRA + LLaMA 3.2)

This project fine-tunes [LLaMA 3.2](https://huggingface.co/meta-llama) using the [Open5e SRD spell](https://api.open5e.com/v1/spells/) dataset and custom homebrew spells to generate **balanced, creative Dungeons & Dragons 5e spells**.  

**LoRA (Low-Rank Adaptation)** is used for efficient training on consumer GPUs, and enforce D&D spell design rules through dataset curation and formatting.

---

## 📁 Project Structure

```
DnD-spell-generator/
├── data/
│   └── homebrew_spells.json        # Custom spells created by community
├── models/
│   └── lora-checkpoints/           # Saved LoRA adapters (gitignored)
├── notebooks/
│   └── main.ipynb                  # Interactive pipeline
├── scripts/
│   ├── create_dataset.py           # Step 1: API fetch & rationales
│   ├── fetch_spells.py             # Step 2: Merging & formatting
│   ├── finetune_spells.py          # Step 3: Training logic
│   └── generate_spells.py          # Step 4: Inference script
├── media/
│   └── demo.mov                    # Demo recording Unity execution
├── .env.example                    # Template for HF and OpenAI keys
├── .gitignore                      # Ignores .env, __pycache__, models/
├── README.md                       # Project documentation
└── requirements.txt                # Dependency list
```
---


## ⚙️ Installation
Clone the repo and install dependencies inside a virtual environment (or conda):

```bash
git clone https://github.com/your-username/DnD-spell-generator.git
cd DnD-spell-generator
```

# Create environment (conda or venv)

```bash
conda create -n dnd-spell python=3.10 -y
conda activate dnd-spell
```

# Install dependencies

```bash
pip install -r requirements.txt
```

You’ll also need:

A Hugging Face account (for LLaMA access)

An OpenAI API key (for generating rationales in create_dataset.py)

Create .env file to set api key

## 🚀 Workflow

The pipeline has 4 stages:

1. Fetch + Rationale (create_dataset.py)

Fetches all SRD spells from Open5e API, generates rationales (via GPT API), and saves them to data/golden_dataset.json.

```bash
cd scripts
python create_dataset.py
```

2. Merge + Format (fetch.py)

Merges SRD + homebrew spells, deduplicates, and formats into LLaMA 3 chat format (data/formatted_dnd_spells.jsonl).

```bash
python fetch_spells.py
```

3. Fine-Tune (finetune.py)

Runs LoRA fine-tuning on the formatted dataset using Hugging Face transformers, trl, and peft.

```bash
python finetune_spells.py
```

4. Generate (generate.py)

Loads the fine-tuned model and generates creative, balanced D&D spells.

```bash
python generate_spells.py
```

## Notebook Mode

You can also run the entire pipeline interactively in Jupyter:

```bash
jupyter notebook notebooks/main.ipynb
```

Or execute directly from the command line:

```bash
jupyter nbconvert --to notebook --execute notebooks/main.ipynb --output notebooks/main-output.ipynb
```
