import argparse
import gradio as gr
import torch
from src.models.inference import LlavaInference
from src.utils.logging import get_logger

# Use "demo" as logger name to clearly indicate inference/demo phase
logger = get_logger("demo")

def main():
    parser = argparse.ArgumentParser(description="LLaVA Gradio Demo")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model (e.g. meta-llama/Meta-Llama-3-8B-Instruct)")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--no_quant", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    logger.info(f"Initializing Inference Engine with base={args.base_model}, adapter={args.adapter}")
    
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    load_in_4bit = not args.no_quant
    
    inference = LlavaInference(
        model_path=args.base_model,
        adapter_path=args.adapter,
        load_in_4bit=load_in_4bit,
        device=device
    )
    
    def predict(image, prompt, temperature, top_p):
        if image is None:
            return "Please upload an image."
        if not prompt:
            return "Please enter a prompt."
            
        logger.info(f"Generating for prompt: {prompt}")
        response = inference.generate(
            image, 
            prompt, 
            temperature=temperature, 
            top_p=top_p
        )
        return response

    with gr.Blocks(title="LLaVA Scalable Tuning Demo") as demo:
        gr.Markdown("# 🌋 LLaVA Scalable Tuning Demo")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")
                text_input = gr.Textbox(label="User Prompt", placeholder="Describe this image...")
                
                with gr.Accordion("Advanced Options", open=False):
                    temp_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, label="Temperature")
                    top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, label="Top-P")
                
                submit_btn = gr.Button("Generate", variant="primary")
                
            with gr.Column():
                output_text = gr.Textbox(label="Assistant Response", lines=10)
        
        submit_btn.click(
            fn=predict,
            inputs=[image_input, text_input, temp_slider, top_p_slider],
            outputs=output_text
        )
        
        gr.Examples(
            examples=[
                ["data/coco/val2017/000000039769.jpg", "What is in this image?"],
            ],
            inputs=[image_input, text_input]
        )

    demo.launch(share=False, server_name="0.0.0.0")

if __name__ == "__main__":
    main()
