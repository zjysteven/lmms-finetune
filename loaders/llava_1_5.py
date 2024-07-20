from typing import Tuple

from transformers import AutoProcessor, LlavaForConditionalGeneration, PreTrainedTokenizer

from . import register_loader
from .base import BaseModelLoader


@register_loader("llava-1.5")
class LLaVA15ModelLoader(BaseModelLoader):
    def load(self) -> Tuple[LlavaForConditionalGeneration, PreTrainedTokenizer, AutoProcessor]:
        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_hf_path, 
            **self.loading_kwargs,
        )
        processor = AutoProcessor.from_pretrained(self.model_hf_path)
        tokenizer = processor.tokenizer
        model.config.hidden_size = model.language_model.config.hidden_size # useful for deepspeed
        return model, tokenizer, processor