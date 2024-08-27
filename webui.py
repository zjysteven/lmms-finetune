import os
import sys
import subprocess
from pathlib import Path
import gradio as gr

from supported_models import MODEL_HF_PATH, MODEL_FAMILIES

def launch_training(
    model_id, data_path, eval_data_path, image_folder, video_folder, num_frames,
    train_vision_encoder, use_vision_lora, train_vision_projector,
    use_lora, q_lora, lora_r, lora_alpha,
    ds_stage, per_device_batch_size, grad_accum, num_epochs,
    lr, model_max_len, num_gpus, use_tf32
):
    # Construct the run_id
    run_id = f"{model_id}_lora-{use_lora}_qlora-{q_lora}"

    # Construct the distributed args
    distributed_args = f"--nnodes=1 --nproc_per_node {num_gpus} --rdzv_backend c10d --rdzv_endpoint localhost:0"

    # Construct the command
    cmd = [
        "torchrun",
        *distributed_args.split(),
        "train.py",
        f"--model_id={model_id}",
        f"--data_path={data_path}",
        f"--eval_data_path={eval_data_path}",
        f"--image_folder={image_folder}",
        f"--video_folder={video_folder}",
        f"--num_frames={num_frames}",
        f"--output_dir=./checkpoints/{run_id}",
        "--report_to=wandb",
        f"--run_name={run_id}",
        f"--deepspeed=./ds_configs/{ds_stage}.json",
        "--bf16=True",
        f"--num_train_epochs={num_epochs}",
        f"--per_device_train_batch_size={per_device_batch_size}",
        f"--per_device_eval_batch_size={per_device_batch_size}",
        f"--gradient_accumulation_steps={grad_accum}",
        "--eval_strategy=epoch",
        "--save_strategy=epoch",
        "--save_total_limit=1",
        f"--learning_rate={lr}",
        "--weight_decay=0.",
        "--warmup_ratio=0.03",
        "--lr_scheduler_type=cosine",
        "--logging_steps=1",
        f"--tf32={use_tf32}",
        f"--model_max_length={model_max_len}",
        "--gradient_checkpointing=True",
        "--dataloader_num_workers=4",
        f"--train_vision_encoder={train_vision_encoder}",
        f"--use_vision_lora={use_vision_lora}",
        f"--train_vision_projector={train_vision_projector}",
        f"--use_lora={use_lora}",
        f"--q_lora={q_lora}",
        f"--lora_r={lora_r}",
        f"--lora_alpha={lora_alpha}",
    ]

    # Run the command
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # Stream the output
    output = ""
    for line in process.stdout:
        output += line
        yield output

    # Wait for the process to complete
    process.wait()

    if process.returncode == 0:
        yield output + "\nTraining completed successfully!"
    else:
        yield output + f"\nTraining failed with return code {process.returncode}"

def create_ui():
    with gr.Blocks(css="#container {max-width: 1600px; margin: auto;}") as ui:
        gr.Markdown("# Training GUI of lmms-finetune", elem_id="title")
        
        with gr.Row(elem_id="container"):
            with gr.Column(scale=1):
                gr.Markdown("## Model")
                model_id = gr.Dropdown(
                    choices=list(MODEL_HF_PATH.keys()),
                    value=list(MODEL_HF_PATH.keys())[0] if MODEL_HF_PATH else None,
                    label="Model ID",
                    info="Select the model to be fine-tuned"
                )
                model_hf_path = gr.Textbox(
                    label="HuggingFace Path",
                    interactive=False,
                    info="Corresponding HuggingFace path"
                )
                
                gr.Markdown("## Data")
                data_path = gr.Textbox(
                    value="./example_data/celeba_image_train.json",
                    label="Training Data Path",
                    info="Path to the training data json file"
                )
                eval_data_path = gr.Textbox(
                    value="./example_data/celeba_image_eval.json",
                    label="Evaluation Data Path",
                    info="Path to the evaluation data json file (optional)"
                )
                image_folder = gr.Textbox(
                    value="./example_data/images",
                    label="Image Folder",
                    info="Path to the image root folder"
                )
                video_folder = gr.Textbox(
                    value="./example_data/videos",
                    label="Video Folder",
                    info="Path to the video root folder"
                )
                num_frames = gr.Number(
                    value=8,
                    label="Number of Frames",
                    info="Frames sampled from each video"
                )
                
                
            
            with gr.Column(scale=1):
                gr.Markdown("## Vision")
                train_vision_encoder = gr.Checkbox(
                    value=False,
                    label="Train Vision Encoder",
                    info="Whether to train the vision encoder"
                )
                use_vision_lora = gr.Checkbox(
                    value=False,
                    label="Use Vision LoRA",
                    info="Whether to use LoRA for vision encoder (only effective when 'Train Vision Encoder' is True)"
                )
                train_vision_projector = gr.Checkbox(
                    value=False,
                    label="Train Vision Projector",
                    info="Whether to train the vision projector (only full finetuning is supported)"
                )
                
                gr.Markdown("## LLM")
                use_lora = gr.Checkbox(
                    value=True,
                    label="Use LoRA",
                    info="Whether to use LoRA for LLM"
                )
                q_lora = gr.Checkbox(
                    value=False,
                    label="Use Q-LoRA",
                    info="Whether to use Q-LoRA for LLM; only effective when 'Use LoRA' is True"
                )
                lora_r = gr.Number(
                    value=8,
                    label="LoRA R",
                    info="The LoRA rank (both LLM and vision encoder)"
                )
                lora_alpha = gr.Number(
                    value=8,
                    label="LoRA Alpha",
                    info="The LoRA alpha (both LLM and vision encoder)"
                )
                
            with gr.Column(scale=1):
                gr.Markdown("## Training")
                ds_stage = gr.Dropdown(
                    ["zero2", "zero3"],
                    value="zero3",
                    label="DeepSpeed Stage",
                    info="DeepSpeed stage; choose between zero2 and zero3"
                )
                per_device_batch_size = gr.Number(
                    value=2,
                    label="Per Device Batch Size",
                    info="Batch size per GPU"
                )
                grad_accum = gr.Number(
                    value=1,
                    label="Gradient Accumulation Steps",
                    info="Number of steps to accumulate gradients"
                )
                num_epochs = gr.Number(
                    value=5,
                    label="Number of Epochs",
                    info="Number of training epochs"
                )
                lr = gr.Number(
                    value=2e-5,
                    label="Learning Rate",
                    info="Learning rate for training"
                )
                model_max_len = gr.Number(
                    value=512,
                    label="Model Max Length",
                    info="Maximum input length of the model"
                )
                num_gpus = gr.Number(
                    value=1,
                    label="Number of GPUs",
                    info="Number of GPUs to use for distributed training"
                )
                use_tf32 = gr.Checkbox(
                    value=True,
                    label="Use TF32",
                    info="Whether to use TF32 precision (for Ampere+ GPUs)"
                )

        train_button = gr.Button("Start Training", variant="primary")
        output = gr.Textbox(label="Training Output", interactive=False)

        def update_hf_path(selected_model):
            return MODEL_HF_PATH.get(selected_model, "")
        
        model_id.change(update_hf_path, inputs=[model_id], outputs=[model_hf_path])

        train_button.click(
            launch_training,
            inputs=[
                model_id, data_path, eval_data_path, image_folder, video_folder, num_frames,
                train_vision_encoder, use_vision_lora, train_vision_projector,
                use_lora, q_lora, lora_r, lora_alpha,
                ds_stage, per_device_batch_size, grad_accum, num_epochs,
                lr, model_max_len, num_gpus, use_tf32
            ],
            outputs=output
        )

    return ui

# Launch the Gradio interface
if __name__ == "__main__":
    ui = create_ui()
    ui.launch(share=True)