#finetune.py

import torch
import os
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # scripts/ → project root
DATA_DIR = PROJECT_ROOT / "data"
MODEL_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "dnd-lora-checkpoint"
MODEL_FINAL_DIR = PROJECT_ROOT / "models" / "dnd-lora-final"
JSONL_PATH = DATA_DIR / "formatted_dnd_spells.jsonl"

def fine_tune(jsonl_path=JSONL_PATH, model_name="meta-llama/Llama-3.2-3b-instruct"):
    """Fine-tunes the language model on the D&D spells dataset."""
    
    print("Loading model and tokenizer...")
    # It is recommended to have a Hugging Face account and be logged in:
    # `huggingface-cli login`
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16, # Use bfloat16 for Ampere GPUs and newer
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading dataset...")
    #dataset = load_dataset("json", data_files=jsonl_path, split="train")
    dataset = load_dataset("json", data_files=str(jsonl_path), split="train")
    dataset = dataset.shuffle(seed=42)

    training_args = TrainingArguments(
        #output_dir=os.path.join("models","dnd-lora-checkpoint"),
        output_dir=str(MODEL_CHECKPOINT_DIR),
        per_device_train_batch_size=1, # Adjust based on your VRAM
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=50,
        learning_rate=2e-4,
        bf16=True, # Recommended for modern GPUs
        # Use a more memory-efficient optimizer
        optim="paged_adamw_8bit",
        report_to="none",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        # ** Key change: Use the 'text' field from our formatted dataset **
        # dataset_text_field="text", 
        # Set a max sequence length to handle long spells and avoid memory errors
        # max_seq_length=2048, 
        # packing=True, # Packs multiple short examples into one sequence for efficiency
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("Saving final model...")
    # Use trainer.save_model to save the final adapter and model configuration
    trainer.save_model(os.path.join(MODEL_FINAL_DIR))
    print("Fine-tuning complete. Model saved to './dnd-lora-final'.")

if __name__ == "__main__":
    fine_tune()
