from typing import Tuple

from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration, PreTrainedTokenizer

from . import register_loader
from .base import BaseModelLoader


@register_loader("llava-next-video")
class LLaVANeXTVideoModelLoader(BaseModelLoader):
    def load(self) -> Tuple[LlavaNextVideoForConditionalGeneration, PreTrainedTokenizer, LlavaNextVideoProcessor]:
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            self.model_hf_path, 
            **self.loading_kwargs,
        )
        processor = LlavaNextVideoProcessor.from_pretrained(self.model_hf_path)
        tokenizer = processor.tokenizer
        model.config.hidden_size = model.language_model.config.hidden_size # useful for deepspeed
        return model, tokenizer, processor