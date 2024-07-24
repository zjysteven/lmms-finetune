import os
os.environ["WANDB_PROJECT"]= "lmms-ft"
from dataclasses import asdict
import logging
from pathlib import Path
from typing import List
import yaml

from accelerate.utils import DistributedType
import torch
import transformers
from transformers import Trainer, deepspeed

from arguments import ModelArguments, DataArguments, TrainingArguments, LoraArguments
from collators import COLLATORS
from datasets import LazySupervisedDataset
from loaders import LOADERS
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, MULTIMODAL_KEYWORDS
)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # dumping arguments
    output_dir = getattr(training_args, 'output_dir', None)
    assert output_dir is not None, "output_dir is required"
    args_dir = Path(output_dir) / "arguments"
    args_dir.mkdir(parents=True, exist_ok=True)
    yaml.dump(asdict(model_args), open(args_dir / "model.yaml", "w"))
    yaml.dump(asdict(data_args), open(args_dir / "data.yaml", "w"))
    yaml.dump(asdict(training_args), open(args_dir / "training.yaml", "w"))
    yaml.dump(asdict(lora_args), open(args_dir / "lora.yaml", "w"))

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    device_map = None
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if int(os.environ.get("WORLD_SIZE", 1)) != 1 else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            raise ValueError("FSDP or ZeRO3 are not incompatible with QLoRA.")

    bnb_config = None
    if lora_args.use_lora and lora_args.q_lora:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4", 
        )
    
    # load model, tokenizer, processor
    rank0_print("Loading model, tokenizer, processor...")
    loader = LOADERS[model_args.model_family_id](
        model_hf_path=model_args.model_name_or_path,
        compute_dtype=compute_dtype,
        bnb_config=bnb_config,
        use_flash_attn=training_args.use_flash_attn,
        device_map=device_map,
    )
    model, tokenizer, processor = loader.load()
    tokenizer.model_max_length = training_args.model_max_length

    # set up lora
    if lora_args.use_lora:
        rank0_print("LoRA enabled...")
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        rank0_print("Adding LoRA adapters...")
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=find_all_linear_names(
                model, 
                model_args.model_family_id, 
                training_args.freeze_multimodal
            ),
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        
        model = get_peft_model(model, lora_config)
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
    else:
        for name, module in model.named_modules():
            if any(mm_keyword in name for mm_keyword in MULTIMODAL_KEYWORDS[model_args.model_family_id]):
                if training_args.freeze_multimodal:
                    rank0_print(f"\tFreezing {name}...")
                    module.requires_grad_(False)
    
    # load data
    rank0_print("Loading data...")
    train_dataset = LazySupervisedDataset(
        data_path=data_args.data_path,
        image_folder=data_args.image_folder,
        video_folder=data_args.video_folder,
        default_num_frames=data_args.default_num_frames,
        model_family_id=model_args.model_family_id,
        user_key=data_args.user_key,
        assistant_key=data_args.assistant_key
    )
    if data_args.eval_data_path:
        eval_dataset = LazySupervisedDataset(
            data_path=data_args.eval_data_path,
            image_folder=data_args.image_folder,
            video_folder=data_args.video_folder,
            default_num_frames=data_args.default_num_frames,
            model_family_id=model_args.model_family_id,
            user_key=data_args.user_key,
            assistant_key=data_args.assistant_key
        )
    else:
        eval_dataset = None

    # data collator
    data_collator = COLLATORS[model_args.model_family_id](
        tokenizer=tokenizer,
        processor=processor,
    )

    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_state()

    if lora_args.use_lora:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), lora_args.lora_bias
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.save_pretrained(output_dir, state_dict=state_dict)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=output_dir)


if __name__ == "__main__":
    train()