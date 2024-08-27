from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoProcessor, BitsAndBytesConfig


class BaseModelLoader(ABC):
    def __init__(
        self, 
        model_hf_path: str,
        model_local_path: str, 
        compute_dtype: torch.dtype,
        bnb_config: Optional[BitsAndBytesConfig] = None,
        use_flash_attn: bool = False,
        device_map: Optional[Union[Dict, str]] = None,
    ) -> None:
        self.model_hf_path = model_hf_path
        self.model_local_path = model_local_path
        self.loading_kwargs = dict(
            torch_dtype=compute_dtype,
            quantization_config=bnb_config,
            device_map=device_map,
        )
        if use_flash_attn:
            self.loading_kwargs["attn_implementation"] = "flash_attention_2"

    @abstractmethod
    def load(self, load_model: bool = True) -> Tuple[
        PreTrainedModel, Union[None, PreTrainedTokenizer], Union[None, AutoProcessor]
    ]: ...