from typing import Dict, Optional, List
from dataclasses import dataclass, field

import transformers

from supported_models import MODEL_HF_PATH, MODEL_FAMILIES


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="llava-1.5-7b")

    def __post_init__(self):
        assert self.model_id in MODEL_HF_PATH, f"Unknown model_id: {self.model_id}"
        self.model_name_or_path: str = MODEL_HF_PATH[self.model_id]
        assert self.model_id in MODEL_FAMILIES, f"Unknown model_id: {self.model_id}"
        self.model_family_id: str = MODEL_FAMILIES[self.model_id]


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data json file."}
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the evaluation data json file."}
    )
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    num_frames: Optional[int] = field(default=8)
    user_key: Optional[str] = field(default="human")
    assistant_key: Optional[str] = field(default="gpt")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_flash_attn: bool = False
    train_vision_encoder: bool = False
    train_vision_projector: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.remove_unused_columns = False


@dataclass
class LoraArguments:
    use_lora: bool = True
    use_vision_lora: bool = True
    q_lora: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"