# 🔬 ChemAssist: QLoRA Fine-Tuned LLM for Chemical Engineering QA

ChemAssist is an end-to-end Generative AI platform that leverages Parameter-Efficient Fine-Tuning (PEFT) to adapt an open-source Large Language Model for high-fidelity, textbook-standard Chemical Engineering question answering. 

By applying **QLoRA (Quantized Low-Rank Adaptation)** to `Qwen2.5-3B-Instruct`, this system injects deep domain expertise spanning thermodynamics, transport phenomena, reaction kinetics, and plant operations while minimizing computational footprints.

---

## 🏗️ System Architecture & Engineering Blueprint

The architecture is explicitly decoupled into independent pipeline modules for data curation, low-precision quantized optimization, quantitative NLP evaluation, and production serving:

```text
  [Raw JSON Seeds] -> [Data Validation & Deduplication] -> [Train/Val Splits]
                                                                  |
  [Qwen2.5-3B-Instruct] -> [4-bit NF4 Quantization] <-------------+ (SFTTrainer)
                                 |
                       [LoRA Target Adapters] -> [Fine-Tuning Loop]
                                 |
           +---------------------+---------------------+
           |                                           |
  [Inference Engine]                          [Metrics Evaluation]
           |                                  (BLEU, ROUGE, BERTScore)
     [Gradio UI App]