"""
QLoRA Training Pipeline for ChemAssist.

Loads an open-source autoregressive LLM in quantized 4-bit precision, configures
LoRA target adapters via PEFT, processes training/validation json datasets,
and runs optimized Supervised Fine-Tuning (SFT).
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
)
from trl import SFTTrainer
from datasets import load_dataset
from typing import Any

def formatting_prompts_func(example: dict) -> Any:
    """
    Formats incoming dataset rows into an explicit text prompt structure for training.
    Returns a list of strings if given a batch, or a single string if given a single example.
    
    Args:
        example: Dictionary containing text records mapped from the dataset.
        
    Returns:
        A string (for single example streaming) or list of strings (for batched data).
    """
    # Check if the inputs are a batch (lists) or a single row (strings)
    if isinstance(example["instruction"], list):
        output_texts = []
        iterations = len(example["instruction"])
        for i in range(iterations):
            instruction = example["instruction"][i]
            input_context = example["input"][i] if "input" in example and example["input"][i] else ""
            output = example["output"][i]
            
            if input_context:
                text = (
                    f"Below is an instruction that describes a task, paired with an input that provides further context.\n"
                    f"Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{instruction}\n\n### Input:\n{input_context}\n\n### Response:\n{output}"
                )
            else:
                text = (
                    f"Below is an instruction that describes a task.\n"
                    f"Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                )
            output_texts.append(text)
        return output_texts
        
    else:
        # Single record execution mode: return a single raw string
        instruction = example["instruction"]
        input_context = example.get("input", "")
        output = example["output"]
        
        if input_context:
            text = (
                f"Below is an instruction that describes a task, paired with an input that provides further context.\n"
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Input:\n{input_context}\n\n### Response:\n{output}"
            )
        else:
            text = (
                f"Below is an instruction that describes a task.\n"
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            )
        return text


def run_training() -> None:
    """Configures the local compute architecture, loads requirements, and triggers QLoRA training."""
    
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    output_dir = "./checkpoints/chass_qlora"
    
    print("[INFO] Preparing target tokenizer configuration...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[INFO] Loading datasets from storage shards...")
    dataset = load_dataset(
        "json",
        data_files={"train": "data/train.json", "validation": "data/val.json"}
    )

    print("[INFO] Initializing 4-bit BitsAndBytes quantization topology...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[INFO] Initializing base model: {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Enable gradient checkpointing and prepare layers for low-precision adapters
    model.config.use_cache = False
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)

    print("[INFO] Configuring Parameter-Efficient LoRA Adapters...")
    # Target projection layers typical to Qwen/Llama attention mechanisms
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    print("[INFO] Structuring Trainer Arguments hyperparameters...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        max_steps=-1,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        fp16=True,
        bf16=False,
        logging_dir="./runs/logs",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )

    print("[INFO] Starting SFTTrainer execution...")
    # Inject maximum token length parameters straight into the arguments payload for TRL compliance
    training_args.max_seq_length = 512

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,  # Updated for current TRL conventions
        formatting_func=formatting_prompts_func,
        args=training_args,
    )

    trainer.train()

    print("[INFO] Saving ultimate optimized PEFT adapters...")
    final_adapter_dir = "./models/chemassist_adapter"
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print(f"[SUCCESS] Pipeline complete. Adapter checkpoints housed safely at: {final_adapter_dir}")


if __name__ == "__main__":
    # Ensure standard CUDA execution settings configuration
    if torch.cuda.is_available():
        run_training()
    else:
        print("[ERROR] CUDA environment unavailable. Training aborting.")
