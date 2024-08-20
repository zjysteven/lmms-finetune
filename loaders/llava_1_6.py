from typing import Tuple

from transformers import AutoProcessor, LlavaNextForConditionalGeneration, PreTrainedTokenizer

from . import register_loader
from .base import BaseModelLoader


@register_loader("llava-1.6")
class LLaVA16ModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True) -> Tuple[LlavaNextForConditionalGeneration, PreTrainedTokenizer, AutoProcessor]:
        if load_model:
            model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_hf_path, 
                **self.loading_kwargs,
            )
            model.config.hidden_size = model.language_model.config.hidden_size # useful for deepspeed
        else:
            model = None

        processor = AutoProcessor.from_pretrained(self.model_hf_path)
        tokenizer = processor.tokenizer
        return model, tokenizer, processor