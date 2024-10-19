from typing import Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from . import register_loader
from .base import BaseModelLoader


@register_loader("qwen-vl")
class QwenVLModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer, None, AutoConfig]:
        self.loading_kwargs["trust_remote_code"] = True
        model = AutoModelForCausalLM.from_pretrained(
            self.model_local_path, 
            **self.loading_kwargs,
        ) if load_model else None
        tokenizer = AutoTokenizer.from_pretrained(self.model_hf_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(self.model_local_path, trust_remote_code=True)
        return model, tokenizer, None, config