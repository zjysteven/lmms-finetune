from typing import Tuple

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, PreTrainedTokenizer

from . import register_loader
from .base import BaseModelLoader


@register_loader("qwen2-vl")
class Qwen2VLModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True) -> Tuple[Qwen2VLForConditionalGeneration, PreTrainedTokenizer, AutoProcessor]:
        if load_model:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_hf_path, 
                **self.loading_kwargs,
            )
            model.config.hidden_size = model.config.hidden_size # useful for deepspeed
        else:
            model = None

        processor = AutoProcessor.from_pretrained(self.model_hf_path)
        tokenizer = processor.tokenizer        
        return model, tokenizer, processor