import os
os.environ["WANDB_PROJECT"]= "lmms-ft"
from dataclasses import asdict
import math
from pathlib import Path
from typing import List, Optional
import yaml

from accelerate.utils import DistributedType
import torch
from torch.utils.data import Sampler
import transformers
from transformers import Trainer, deepspeed
from transformers.trainer import has_length

from arguments import ModelArguments, DataArguments, TrainingArguments, LoraArguments
from collators import COLLATORS
from datasets import LazySupervisedDataset
from loaders import LOADERS
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3, MULTIMODAL_KEYWORDS
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class NoTextOnlyBatchSampler(Sampler):
    r"""
    Sampler that tries its best to sample batches such that no batch has only 
    text (unimodal) data. This is necessary for training with deepspeed. 
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        is_text_only: Optional[List[bool]] = None,
        generator=None,
    ):
        if is_text_only is None:
            raise ValueError("`is_text_only` must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.is_text_only = is_text_only
        self.generator = generator
        self.mega_batch_size = batch_size * world_size

    def __len__(self):
        return len(self.is_text_only)

    def __iter__(self):
        # mm: multimodal, entry that has both text and image/video
        # uni: unimodal, entry that has only text
        mm_indices = [i for i, is_text_only in enumerate(self.is_text_only) if not is_text_only]
        uni_indices = [i for i, is_text_only in enumerate(self.is_text_only) if is_text_only]

        num_batches = math.ceil((len(mm_indices) + len(uni_indices)) / self.mega_batch_size)
        if len(mm_indices) < num_batches:
            raise ValueError(
                "Not enough multimodal data in the dataset, or the batch size is too small. " 
                "There will be at least one batch that is text-only, which doesn't work with deepspeed."
            )

        # shuffle indices
        mm_indices = [mm_indices[i] for i in torch.randperm(len(mm_indices), generator=None).tolist()]
        uni_indices = [uni_indices[i] for i in torch.randperm(len(uni_indices), generator=None).tolist()]

        # distribute indices into batches
        num_uni_indices_in_mega_batch = [len(uni_indices) // num_batches] * num_batches
        for i in range(len(uni_indices) % num_batches):
            num_uni_indices_in_mega_batch[i] += 1
        
        mega_batches = []
        cur_uni_index = 0
        cur_mm_index = 0
        for i, num_uni_indices in enumerate(num_uni_indices_in_mega_batch):
            mega_batch = []
            mega_batch.extend(uni_indices[cur_uni_index:cur_uni_index + num_uni_indices])
            cur_uni_index += num_uni_indices
            assert len(mega_batch) < self.mega_batch_size

            if i < num_batches - 1:
                increment = self.mega_batch_size - len(mega_batch)
                mega_batch.extend(
                    mm_indices[cur_mm_index:cur_mm_index + increment]
                )
                cur_mm_index += increment
            else: # last batch
                mega_batch.extend(mm_indices[cur_mm_index:])
                assert len(mega_batch) <= self.mega_batch_size, "Last batch is too big."
            
            mega_batches.append(mega_batch)
        
        mega_batch_indices = torch.randperm(len(mega_batches), generator=self.generator)
        mega_batches = [mega_batches[i] for i in mega_batch_indices]
        indices = [i for mega_batch in mega_batches for i in mega_batch]
        return iter(indices)


class TrainerWithCustomSampler(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        is_text_only = self.train_dataset.is_text_only
        return NoTextOnlyBatchSampler(
            self.args.train_batch_size,
            world_size=self.args.world_size * self.args.gradient_accumulation_steps,
            is_text_only=is_text_only,
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

    if training_args.vision_encoder_training == "lora":
        vision_target_modules = find_all_linear_names(
            model.vision_tower,
            model_args.model_family_id,
            False
        )
        vision_lora_config = LoraConfig(
            r=lora_args.vision_lora_r, 
            lora_alpha=lora_args.vision_lora_alpha,
            target_modules=vision_target_modules,
            lora_dropout=lora_args.vision_lora_dropout,
            bias=lora_args.vision_lora_bias,
            task_type="CAUSAL_LM",
        )
    # set up lora for lm and vision encoder
    if lora_args.use_lora:
        rank0_print("LoRA enabled...")
        # Find target modules for both language model and vision encoder
        lm_target_modules = find_all_linear_names(
        model, 
        model_args.model_family_id, 
        training_args.freeze_multimodal
        )
        lm_lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lm_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
            
        # Apply LoRA to LM
        model = get_peft_model(model, lm_lora_config)
        
        # Apply LoRA to vision encoder if not freezing multimodal
        if training_args.vision_encoder_training == "lora":
            rank0_print("Applying LoRA to vision encoder...")
            model.vision_tower = get_peft_model(model.vision_tower, vision_lora_config)
        elif training_args.vision_encoder_training == "freeze":
            rank0_print("Freezing vision encoder...")
            for param in model.vision_tower.parameters():
                param.requires_grad = False
        elif training_args.vision_encoder_training == "full":
            rank0_print("Fully training vision encoder...")
            for param in model.vision_tower.parameters():
                param.requires_grad = True
        
        if training_args.vision_projector_training == "freeze":
            rank0_print("Freezing vision projector...")
            for param in model.multi_modal_projector.parameters():
                param.requires_grad = False
        elif training_args.vision_projector_training == "full":
            rank0_print("Fully training vision projector...")
            for param in model.multi_modal_projector.parameters():
                param.requires_grad = True
    
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
        training_args.eval_strategy = "no"

    # data collator
    data_collator = COLLATORS[model_args.model_family_id](
        tokenizer=tokenizer,
        processor=processor,
    )

    # trainer
    trainer = TrainerWithCustomSampler(
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

    if training_args.local_rank == 0 or training_args.local_rank == -1:

        # Save vision encoder
        if training_args.vision_encoder_training == "lora":
            vision_encoder_path = os.path.join(output_dir, "vision_encoder")
            os.makedirs(vision_encoder_path, exist_ok=True)
            vision_encoder_state_dict = get_peft_state_maybe_zero_3(
                model.vision_tower.named_parameters(), lora_args.vision_lora_bias
            )
            model.vision_tower.save_pretrained(vision_encoder_path, state_dict=vision_encoder_state_dict)
        elif training_args.vision_encoder_training == "full":
            vision_encoder_path = os.path.join(output_dir, "vision_encoder")
            os.makedirs(vision_encoder_path, exist_ok=True)
            model.vision_tower.save_pretrained(vision_encoder_path)
        
        
        # Save vision projector if it was trained
        if training_args.vision_projector_training == "full":
            vision_projector_path = os.path.join(output_dir, "vision_projector")
            os.makedirs(vision_projector_path, exist_ok=True)
            torch.save(model.multi_modal_projector.state_dict(), os.path.join(vision_projector_path, "projector.pth"))
    


if __name__ == "__main__":
    train()