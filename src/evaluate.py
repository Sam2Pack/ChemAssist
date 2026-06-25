"""
Evaluation Suite for ChemAssist.

Loads test datasets, conducts comparative batch inference across the Base Model 
vs. the Fine-Tuned PEFT Model, and computes BLEU, ROUGE, and BERTScore metrics.
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import evaluate


class ModelEvaluator:
    """Manages quantitative verification pipelines using standardized text metrics."""

    def __init__(self, base_model_id: str = "Qwen/Qwen2.5-3B-Instruct", adapter_dir: str = "./models/chemassist_adapter") -> None:
        """
        Initializes evaluation configurations and metric loaders.
        
        Args:
            base_model_id: HuggingFace model hub path string.
            adapter_dir: Path to the locally saved LoRA adapter directory.
        """
        self.base_model_id = base_model_id
        self.adapter_dir = adapter_dir
        
        print("[INFO] Loading evaluation evaluation metrics...")
        self.bleu = evaluate.load("sacrebleu")
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")
        
        print("[INFO] Setting up 4-bit Quantization configs for evaluation...")
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_response(self, model: Any, instruction: str) -> str:
        """Runs specialized greedy sequence decoding for evaluation."""
        prompt = (
            f"Below is an instruction that describes a task.\n"
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Response:\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False  # Deterministic generation for metrics calculation
            )
            
        generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return generated_text.strip()

    def run_evaluation_suite(self, val_data_path: str = "data/val.json", sample_limit: int = 5) -> None:
        """Loads verification instances, performs inference, and prints structural score metrics."""
        if not os.path.exists(val_data_path):
            print(f"[ERROR] Evaluation aborted. Target validation file path not found: {val_data_path}")
            return

        with open(val_data_path, "r", encoding="utf-8") as f:
            records = json.load(f)[:sample_limit]

        instructions = [r["instruction"] for r in records]
        references = [r["output"] for r in records]

        # 1. Base Model Inference Segment
        print(f"\n[INFO] Loading Un-adapted Base Model: {self.base_model_id}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("[INFO] Generating predictions from base model...")
        base_predictions = [self.generate_response(base_model, inst) for inst in instructions]
        
        # Clean VRAM footprints completely to prevent OOM
        del base_model
        torch.cuda.empty_cache()

        # 2. Fine-Tuned Model Inference Segment
        print(f"\n[INFO] Loading Fine-Tuned Model Adapters from {self.adapter_dir}...")
        raw_base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        peft_model = PeftModel.from_pretrained(raw_base_model, self.adapter_dir)
        
        print("[INFO] Generating predictions from fine-tuned model...")
        ft_predictions = [self.generate_response(peft_model, inst) for inst in instructions]
        
        del peft_model, raw_base_model
        torch.cuda.empty_cache()

        # 3. Compute Metrics
        print("\n==================================================")
        print("           METRICS EVALUATION REPORT              ")
        print("==================================================")
        
        for name, preds in [("Base Model", base_predictions), ("Fine-Tuned Model", ft_predictions)]:
            # Formulate SacreBLEU formatting arrays
            bleu_res = self.bleu.compute(predictions=preds, references=[[r] for r in references])
            rouge_res = self.rouge.compute(predictions=preds, references=references)
            bert_res = self.bertscore.compute(predictions=preds, references=references, lang="en")

            print(f"\n📈 Performance Statistics for: {name}")
            print(f"  - BLEU Score : {bleu_res['score']:.4f}")
            print(f"  - ROUGE-L    : {rouge_res['rougeL']:.4f}")
            print(f"  - BERTScore  : {sum(bert_res['f1'])/len(bert_res['f1']):.4f}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        evaluator = ModelEvaluator()
        # Evaluate performance on a lightweight slice of validation records
        evaluator.run_evaluation_suite(sample_limit=5)
    else:
        print("[ERROR] CUDA required for comparative model evaluation execution.")