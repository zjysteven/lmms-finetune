import os
import sys
import subprocess
from pathlib import Path
import gradio as gr

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
    with gr.Blocks() as ui:
        gr.Markdown("# Training GUI of lmms-finetune")
        
        with gr.Tab("Model"):
            model_id = gr.Textbox(value="llava-next-video-7b", label="Model ID")
        
        with gr.Tab("Data"):
            data_path = gr.Textbox(value="./example_data/ego4d_video_train.json", label="Training Data Path")
            eval_data_path = gr.Textbox(value="./example_data/ego4d_video_eval.json", label="Evaluation Data Path")
            image_folder = gr.Textbox(value="./example_data/images", label="Image Folder")
            video_folder = gr.Textbox(value="./example_data/videos", label="Video Folder")
            num_frames = gr.Number(value=8, label="Number of Frames")
        
        with gr.Tab("Vision"):
            train_vision_encoder = gr.Checkbox(value=False, label="Train Vision Encoder")
            use_vision_lora = gr.Checkbox(value=False, label="Use Vision LoRA")
            train_vision_projector = gr.Checkbox(value=False, label="Train Vision Projector")
        
        with gr.Tab("LLM"):
            use_lora = gr.Checkbox(value=True, label="Use LoRA")
            q_lora = gr.Checkbox(value=False, label="Use Q-LoRA")
            lora_r = gr.Number(value=8, label="LoRA R")
            lora_alpha = gr.Number(value=8, label="LoRA Alpha")
        
        with gr.Tab("Training"):
            ds_stage = gr.Dropdown(["zero2", "zero3"], value="zero3", label="DeepSpeed Stage")
            per_device_batch_size = gr.Number(value=2, label="Per Device Batch Size")
            grad_accum = gr.Number(value=1, label="Gradient Accumulation Steps")
            num_epochs = gr.Number(value=5, label="Number of Epochs")
            lr = gr.Number(value=2e-5, label="Learning Rate")
            model_max_len = gr.Number(value=512, label="Model Max Length")
            num_gpus = gr.Number(value=1, label="Number of GPUs")
            use_tf32 = gr.Checkbox(value=False, label="Use TF32")
        
        train_button = gr.Button("Start Training")
        output = gr.Textbox(label="Training Output", interactive=False)
        
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
    ui.launch()