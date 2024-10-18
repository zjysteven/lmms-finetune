from typing import Tuple

from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration, PreTrainedTokenizer, AutoConfig

from . import register_loader
from .base import BaseModelLoader


@register_loader("llava-next-video")
class LLaVANeXTVideoModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True) -> Tuple[LlavaNextVideoForConditionalGeneration, PreTrainedTokenizer, LlavaNextVideoProcessor, AutoConfig]:
        if load_model:
            model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                self.model_local_path, 
                **self.loading_kwargs,
            )
            model.config.hidden_size = model.language_model.config.hidden_size # useful for deepspeed
        else:
            model = None

        processor = LlavaNextVideoProcessor.from_pretrained(self.model_hf_path, add_eos_token=True)
        tokenizer = processor.tokenizer
        config = AutoConfig.from_pretrained(self.model_local_path)
        return model, tokenizer, processor, config