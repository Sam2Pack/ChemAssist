"""
Inference Engine Module for ChemAssist.

Loads the fine-tuned QLoRA adapter layers over the quantized base model
and provides a clean interface for executing standalone domain-specific queries.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import Optional


class ChemAssistEngine:
    """Handles runtime initialization and response generation for the fine-tuned model."""

    def __init__(self, base_model_id: str = "Qwen/Qwen2.5-3B-Instruct", adapter_dir: str = "./models/chemassist_adapter") -> None:
        """
        Initializes the inference environment, loading weights into memory.
        
        Args:
            base_model_id: HuggingFace model hub path string.
            adapter_dir: Path to the locally saved LoRA adapter directory.
        """
        self.base_model_id = base_model_id
        self.adapter_dir = adapter_dir
        
        # Verify adapter footprint exists before attempting memory allocation
        if not os.path.exists(self.adapter_dir):
            raise FileNotFoundError(
                f"Trained adapter weights missing at: {self.adapter_dir}. "
                f"Please execute 'src/train.py' successfully first."
            )

        print("[INFO] Setting up 4-bit runtime quantization layout...")
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        print(f"[INFO] Initializing base tokenization streams...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[INFO] Loading underlying base architecture weights...")
        raw_base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        print(f"[INFO] Merging model runtime with LoRA adapters...")
        self.model = PeftModel.from_pretrained(raw_base_model, self.adapter_dir)
        self.model.eval()  # Disables dropout matrices for static production inference
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def answer_question(self, instruction: str, input_context: Optional[str] = None, temperature: float = 0.3, top_p: float = 0.9) -> str:
        """
        Processes text inputs, runs text generation, and parses the output response.
        
        Args:
            instruction: The primary prompt or question.
            input_context: Supplementary context or problem constraints (optional).
            temperature: Sampling temperature variance parameter.
            top_p: Nucleus filtering threshold.
            
        Returns:
            The pure decoded response text string.
        """
        # Maintain perfect alignment with formatting paradigms configured at training time
        if input_context:
            prompt = (
                f"Below is an instruction that describes a task, paired with an input that provides further context.\n"
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Input:\n{input_context}\n\n### Response:\n"
            )
        else:
            prompt = (
                f"Below is an instruction that describes a task.\n"
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n### Response:\n"
            )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                top_p=top_p,
                do_sample=True if temperature > 0.0 else False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Slice away the original prompt matrix prefix to isolate only the new answer tokens
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return decoded_text.strip()


if __name__ == "__main__":
    if torch.cuda.is_available():
        # Quick execution test to verify end-to-end interface health
        try:
            engine = ChemAssistEngine()
            test_query = "Define cavitation in centrifugal pumps and outline steps to prevent its occurrence."
            print(f"\n[QUERY]: {test_query}\n")
            response = engine.answer_question(instruction=test_query)
            print(f"[CHEMASSIST RESPONSE]:\n{response}")
        except Exception as e:
            print(f"[Initialization Error/Warning]: {e}")
    else:
        print("[ERROR] CUDA architecture required to safely perform 4-bit adapter execution.")