from typing import Tuple

from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, AutoConfig

from . import register_loader
from .base import BaseModelLoader


@register_loader("glm-4v")
class GLM4VModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer, AutoProcessor, AutoConfig]:
        if load_model:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_local_path,
                trust_remote_code=True,
                **self.loading_kwargs,
            )
            model.config.hidden_size = model.config.hidden_size # useful for deepspeed
        else:
            model = None

        processor = AutoProcessor.from_pretrained(self.model_hf_path, trust_remote_code=True,)
        tokenizer = AutoTokenizer.from_pretrained(self.model_hf_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(self.model_local_path, trust_remote_code=True,)
        return model, tokenizer, processor, config