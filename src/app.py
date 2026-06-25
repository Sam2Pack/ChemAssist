"""
Gradio Web Application Interface for ChemAssist.

Launches a responsive, styled local or cloud web dashboard enabling 
interactive question-answering using the fine-tuned QLoRA chemical engineering model.
"""

import sys
import os
import gradio as gr

# Force the project root directory onto the system path to allow smooth module resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference import ChemAssistEngine

# Global placeholder for the inference model wrapper
engine = None


def initialize_engine() -> bool:
    """
    Safely allocates weight memory and initializes the inference container.
    
    Returns:
        Boolean mapping state confirmation.
    """
    global engine
    if engine is not None:
        return True
    try:
        engine = ChemAssistEngine()
        return True
    except Exception as e:
        print(f"[CRITICAL] App engine failed to mount weights: {e}")
        return False


def process_query(instruction: str, context: str, temperature: float, top_p: float) -> str:
    """
    Binds the front-end parameters directly to the underlying model inference routine.
    
    Args:
        instruction: Main text prompt question textbox string.
        context: Secondary input textbox block string.
        temperature: Decimal scaling factor adjusting prediction variance.
        top_p: Nucleus configuration range constraint.
        
    Returns:
        The decoded raw textual response or descriptive runtime error logs.
    """
    if not instruction.strip():
        return "⚠️ Error: The instruction field cannot be empty. Please input a valid question."

    global engine
    if engine is None:
        success = initialize_engine()
        if not success:
            return (
                "❌ Error: Model weights could not be loaded into memory.\n"
                "Please verify that 'src/train.py' completed successfully and created the adapter files at "
                "'./models/chemassist_adapter'."
            )

    try:
        # Route processing through our standalone class instance
        response = engine.answer_question(
            instruction=instruction,
            input_context=context if context.strip() else None,
            temperature=temperature,
            top_p=top_p
        )
        return response
    except Exception as e:
        return f"❌ An unexpected structural exception occurred during inference processing:\n{str(e)}"


def build_app() -> gr.Blocks:
    """
    Assembles layout rows, colors, documentation descriptions, and event hooks for the UI.
    
    Returns:
        The fully populated Gradio Blocks deployment object.
    """
    # Use clean, minimalist CSS parameters to make the app portfolio-ready
    custom_theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="slate",
        neutral_hue="neutral"
    )

    with gr.Blocks(theme=custom_theme, title="ChemAssist Professional QA Portal") as demo:
        gr.Markdown(
            """
            # 🔬 ChemAssist: Specialised AI for Chemical Engineering
            ### Parameter-Efficient QLoRA Fine-Tuning Portfolio Project
            
            This professional assistant is built upon a fine-tuned **Qwen2.5-3B-Instruct** foundation architecture, optimized using 4-bit quantized low-rank adaptation adapters to output textbook-standard answers for core Chemical Engineering disciplines.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📥 Query Parameters & Inputs")
                instruction_box = gr.Textbox(
                    label="Instruction / Question",
                    placeholder="Enter your Chemical Engineering question here...",
                    lines=4
                )
                context_box = gr.Textbox(
                    label="Supplementary Input Context (Optional)",
                    placeholder="Enter operational boundaries, raw experimental values, specific test parameters...",
                    lines=2
                )
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Inputs", variant="secondary")
                    submit_btn = gr.Button("⚡ Generate Response", variant="primary")
            
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Generation Configurations")
                temp_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                    label="Temperature (Lower values are more precise)"
                )
                top_p_slider = gr.Slider(
                    minimum=0.5, maximum=1.0, value=0.9, step=0.05,
                    label="Top-P (Nucleus Filtering)"
                )
                
                gr.Markdown(
                    """
                    **📋 Architecture Information:**
                    - **Base Weights:** Qwen2.5-3B-Instruct
                    - **Quantization Layer:** BitsAndBytes NF4 4-bit
                    - **Adapter Pattern:** PEFT LoRA (Rank 16, Alpha 32)
                    - **Target Modality:** Multi-Disciplinary Text & Math
                    """
                )

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### 📤 Model Response Output")
                output_box = gr.Textbox(
                    label="ChemAssist Textbook-Style Output Response",
                    placeholder="The engine's analysis will render here...",
                    lines=12,
                    show_copy_button=True
                )

        # Prepopulate clear example variations to let portfolio visitors test it instantly
        gr.Examples(
            examples=[
                [
                    "Calculate the log mean temperature difference (LMTD) for a counter-current heat exchanger where the hot fluid enters at 373 K and leaves at 323 K, while the cold fluid enters at 283 K and leaves at 303 K.",
                    "", 0.1, 0.9
                ],
                [
                    "Explain the physical significance of the Thiele Modulus in heterogeneous catalysis.",
                    "", 0.3, 0.9
                ],
                [
                    "Define cavitation in centrifugal pumps and outline steps to prevent its occurrence.",
                    "", 0.3, 0.9
                ]
            ],
            inputs=[instruction_box, context_box, temp_slider, top_p_slider],
            outputs=output_box,
            fn=process_query,
            cache_examples=False
        )

        # Wire event triggers sequentially
        submit_btn.click(
            fn=process_query,
            inputs=[instruction_box, context_box, temp_slider, top_p_slider],
            outputs=output_box
        )
        
        clear_btn.click(
            fn=lambda: ("", "", 0.3, 0.9, ""),
            inputs=None,
            outputs=[instruction_box, context_box, temp_slider, top_p_slider, output_box]
        )

    return demo


if __name__ == "__main__":
    # Attempt parsing configuration on launch to flag missing adapters right away
    if not os.path.exists("./models/chemassist_adapter"):
        print(
            "[WARN] Local adapter files not found. The app will launch, but it will require "
            "the fine-tuned adapter directory to be generated before processing user queries."
        )
    
    app = build_app()
    # Launch on a standard port; share=True creates a public URL for deployment demonstration
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)