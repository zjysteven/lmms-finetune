from typing import Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoConfig

from . import register_loader
from .base import BaseModelLoader


@register_loader("phi3-v")
class Phi3VModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoConfig]:
        self.loading_kwargs["trust_remote_code"] = True
        model = AutoModelForCausalLM.from_pretrained(
            self.model_local_path, 
            **self.loading_kwargs,
        ) if load_model else None
        processor = AutoProcessor.from_pretrained(self.model_hf_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        config = AutoConfig.from_pretrained(self.model_local_path)
        return model, tokenizer, processor, config