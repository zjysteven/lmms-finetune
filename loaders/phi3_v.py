from typing import Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

from . import register_loader
from .base import BaseModelLoader


@register_loader("phi3-v")
class Phi3VModelLoader(BaseModelLoader):
    def load(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer, AutoProcessor]:
        self.loading_kwargs["trust_remote_code"] = True
        model = AutoModelForCausalLM.from_pretrained(
            self.model_hf_path, 
            **self.loading_kwargs,
        )
        processor = AutoProcessor.from_pretrained(self.model_hf_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        return model, tokenizer, processor