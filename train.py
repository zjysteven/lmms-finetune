import os
os.environ["WANDB_PROJECT"]= "lmms-ft"
from dataclasses import asdict
import math
from pathlib import Path
from typing import List, Optional
import yaml

from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from torch.utils.data import Sampler
import transformers
from transformers import Trainer, deepspeed
from transformers.trainer import has_length

from arguments import ModelArguments, DataArguments, TrainingArguments, LoraArguments
from collators import COLLATORS
from datasets import LazySupervisedDataset
from loaders import LOADERS
from supported_models import MODULE_KEYWORDS
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3
)


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

    # llm quantization config (for q-lora)
    bnb_config = None
    if lora_args.use_lora and lora_args.q_lora:
        from transformers import BitsAndBytesConfig
        rank0_print("Quantization for LLM enabled...")
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

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # freeze certain params
    vision_encoder_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_encoder"]
    if not training_args.train_vision_encoder:
        rank0_print(f"Vision encoder is freezed... including:")
        for module in vision_encoder_keys:
            rank0_print(f"\t{module}")
            eval(f"model.{module}").requires_grad_(False)

    vision_projector_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_projector"]
    if not training_args.train_vision_projector:
        rank0_print(f"Vision projector is freezed... including:")
        for module in vision_projector_keys:
            rank0_print(f"\t{module}")
            eval(f"model.{module}").requires_grad_(False)

    # other components preparation (e.g., image_newline, vision_resampler)
    # we will just freeze these
    if "others" in MODULE_KEYWORDS[model_args.model_family_id]:
        rank0_print(f"Other multimodal component is freezed... including:")
        for other_key in MODULE_KEYWORDS[model_args.model_family_id]["others"]:
            rank0_print(f"\t{other_key}")
            eval(f"model.{other_key}").requires_grad_(False)

    # lora preparation
    llm_keys = MODULE_KEYWORDS[model_args.model_family_id]["llm"]
    if not (lora_args.use_lora or (training_args.train_vision_encoder and lora_args.use_vision_lora)):
        rank0_print("No LoRA enabled...")        
    else:
        named_modules = {n: m for n, m in model.named_modules()}
        lora_modules = []
        full_modules = []

        if training_args.train_vision_encoder and lora_args.use_vision_lora:
            rank0_print("LoRA for vision encoder enabled...")
            lora_modules.extend(find_all_linear_names(named_modules, vision_encoder_keys))
        elif training_args.train_vision_encoder:
            rank0_print("Vision encoder will be fully trained...")
            full_modules.extend(vision_encoder_keys)
        
        if lora_args.use_lora:
            rank0_print("LoRA for LLM enabled...")
            lora_modules.extend(find_all_linear_names(named_modules, llm_keys))
        else:
            rank0_print("LLM will be fully trained...")
            full_modules.extend(llm_keys)
        
        if training_args.train_vision_projector:
            rank0_print("Vision projector will be fully trained...")
            full_modules.extend(vision_projector_keys)
        
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_modules,
            modules_to_save=full_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
            
        model = get_peft_model(model, lora_config)
        
    # print trainable parameters for inspection
    rank0_print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(f"\t{name}")

    # load data
    rank0_print("Loading data...")
    train_dataset = LazySupervisedDataset(
        data_path=data_args.data_path,
        image_folder=data_args.image_folder,
        video_folder=data_args.video_folder,
        num_frames=data_args.num_frames,
        model_family_id=model_args.model_family_id,
        user_key=data_args.user_key,
        assistant_key=data_args.assistant_key
    )
    if data_args.eval_data_path:
        eval_dataset = LazySupervisedDataset(
            data_path=data_args.eval_data_path,
            image_folder=data_args.image_folder,
            video_folder=data_args.video_folder,
            num_frames=data_args.num_frames,
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

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=output_dir)
    

if __name__ == "__main__":
    train()