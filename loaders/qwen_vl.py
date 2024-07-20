from typing import Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

from . import register_loader
from .base import BaseModelLoader


@register_loader("qwen-vl")
class QwenVLModelLoader(BaseModelLoader):
    def load(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer, None]:
        self.loading_kwargs["trust_remote_code"] = True
        model = AutoModelForCausalLM.from_pretrained(
            self.model_hf_path, 
            **self.loading_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_hf_path, trust_remote_code=True)
        return model, tokenizer, None