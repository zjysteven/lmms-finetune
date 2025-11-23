from typing import Tuple

from transformers import AutoProcessor, AutoModel, AutoTokenizer, PreTrainedTokenizer, AutoConfig

from . import register_loader
from .base import BaseModelLoader


IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'


@register_loader("internvl2")
class InternVL2ModelLoader(BaseModelLoader):
    def load(self, load_model: bool = True) -> Tuple[AutoModel, PreTrainedTokenizer, AutoProcessor, AutoConfig]:
        if load_model:
            model = AutoModel.from_pretrained(
                self.model_local_path,
                trust_remote_code=True,
                **self.loading_kwargs,
            )
            model.config.hidden_size = model.config.llm_config.hidden_size # useful for deepspeed
        else:
            model = None

        processor = AutoProcessor.from_pretrained(self.model_hf_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_hf_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(self.model_local_path, trust_remote_code=True)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_context_token_id = img_context_token_id

        return model, tokenizer, processor, config


if __name__ == "__main__":
    import torch
    loader = InternVL2ModelLoader(
        model_hf_path="OpenGVLab/InternVL2-8B",
        model_local_path="/aiarena/group/gmgroup/hongyq/models/OpenGVLab/InternVL2-8B",
        compute_dtype=torch.bfloat16,
    )
    model, tokenizer, processor, config = loader.load()