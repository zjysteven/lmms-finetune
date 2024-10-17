from typing import Tuple

from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, PreTrainedTokenizer, AutoConfig

from . import register_loader
from .base import BaseModelLoader


@register_loader("llava-onevision")
class LLaVAOnevisionModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True) -> Tuple[LlavaOnevisionForConditionalGeneration, PreTrainedTokenizer, AutoProcessor, AutoConfig]:
        if load_model:
            model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                self.model_local_path, 
                **self.loading_kwargs,
            )
            model.config.hidden_size = model.language_model.config.hidden_size # useful for deepspeed
        else:
            model = None

        processor = AutoProcessor.from_pretrained(self.model_hf_path)
        tokenizer = processor.tokenizer
        config = AutoConfig.from_pretrained(self.model_local_path)
        return model, tokenizer, processor, config