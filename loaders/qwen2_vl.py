from typing import Tuple

from transformers import AutoConfig, AutoProcessor, Qwen2VLForConditionalGeneration, PreTrainedTokenizer

from . import register_loader, rank0_print
from .base import BaseModelLoader

@register_loader("qwen2-vl")
class Qwen2VLModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True) -> Tuple[Qwen2VLForConditionalGeneration, PreTrainedTokenizer, AutoProcessor, AutoConfig]:
        if load_model:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_local_path, 
                **self.loading_kwargs,
            )
            # The default range for the number of visual tokens per image in the model is 4-16384.
            # 如果图片token数超过了16384, 就会对图片采用保持长宽比的resize策略
            if self.model_max_length is None:
                rank0_print("Using default processor for Qwen2-VL model")
                min_pixels = 4 * 28 * 28
                max_pixels = 16384 * 28 * 28
            elif self.model_max_length <= 1000:
                raise ValueError("The model_max_length should be greater than 1000")
            else:
                min_pixels = 4 * 28 * 28
                max_pixels = (self.model_max_length - 1000) * 28 * 28
                rank0_print("Using custom processor of max_pixels {} for Qwen2-VL model".format(max_pixels))
            model.config.hidden_size = model.config.hidden_size # useful for deepspeed
        else:
            model = None
        config = AutoConfig.from_pretrained(self.model_local_path)
        processor = AutoProcessor.from_pretrained(self.model_hf_path, min_pixels=min_pixels, max_pixels=max_pixels)
        tokenizer = processor.tokenizer        
        return model, tokenizer, processor, config