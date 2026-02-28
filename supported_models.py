from typing import Dict, List
from collections import OrderedDict

from collators import COLLATORS
from datasets import TO_LOAD_IMAGE
from loaders import LOADERS


MODULE_KEYWORDS: Dict[str, Dict[str, List]] = {
    "llava-1.5": {
        "vision_encoder": ["vision_tower"],
        "vision_projector": ["multi_modal_projector"],
        "llm": ["language_model"]
    },
    "llava-1.6": {
        "vision_encoder": ["vision_tower"],
        "vision_projector": ["multi_modal_projector"],
        "llm": ["language_model"],
        "others": ["image_newline"]
    },
    "llava-interleave": {
        "vision_encoder": ["vision_tower"],
        "vision_projector": ["multi_modal_projector"],
        "llm": ["language_model"]
    },
    "llava-next-video": {
        "vision_encoder": ["vision_tower"],
        "vision_projector": ["multi_modal_projector"],
        "others": ["image_newline", "vision_resampler"],
        "llm": ["language_model"]
    },
    "llava-onevision": {
        "vision_encoder": ["vision_tower"],
        "vision_projector": ["multi_modal_projector"],
        "llm": ["language_model"],
        "others": ["image_newline"]
    },
    "phi3-v": {
        "vision_encoder": ["model.vision_embed_tokens.img_processor"],
        "vision_projector": ["model.vision_embed_tokens.img_projection"],
        "others": ["model.vision_embed_tokens.glb_GN", "model.vision_embed_tokens.sub_GN"],
        "llm": ["model.embed_tokens", "model.layers", "model.norm", "lm_head"]
    },
    "qwen-vl": {
        "vision_encoder": ["transformer.visual.conv1", "transformer.visual.positional_embedding", "transformer.visual.ln_pre", "transformer.visual.transformer", "transformer.visual.ln_post", "transformer.visual.proj"],
        "vision_projector": ["transformer.visual.attn_pool"],
        "llm": ["transformer.wte", "transformer.rotary_emb", "transformer.h", "transformer.ln_f", "lm_head"]
    },
    "qwen2-vl": {
        "vision_encoder": ["visual.patch_embed", "visual.rotary_pos_emb", "visual.blocks"],
        "vision_projector": ["visual.merger"],
        "llm": ["model"]
    },
    "qwen2.5-vl": {
        "vision_encoder": ["visual.patch_embed", "visual.rotary_pos_emb", "visual.blocks"],
        "vision_projector": ["visual.merger"],
        "llm": ["model"]
    },
    "qwen3-vl": {
        "vision_encoder": ["model.visual.patch_embed", "model.visual.pos_embed", "model.visual.blocks"],
        "vision_projector": ["model.visual.merger", "model.visual.deepstack_merger_list"],
        "llm": ["model.language_model"]
    },
    "llama-3.2-vision": {
        "vision_encoder": ["vision_model"],
        "vision_projector": ["multi_modal_projector"],
        "llm": ["language_model"]
    },
    "internvl2": {
        "vision_encoder": ["vision_model"],
        "vision_projector": ["mlp1"],
        "llm": ["language_model"]
    },
    "internvl2.5": {
        "vision_encoder": ["vision_model"],
        "vision_projector": ["mlp1"],
        "llm": ["language_model"]
    },
    "idefics2": {
        "vision_encoder": ["model.vision_model"],
        "vision_projector": ["model.connector"],
        "llm": ["model.text_model", "lm_head"]
    },
    "idefics3": {
        "vision_encoder": ["model.vision_model"],
        "vision_projector": ["model.connector"],
        "llm": ["model.text_model", "lm_head"]
    },
    "glm-4v": {
        "vision_encoder": ["transformer.vision.patch_embedding", "transformer.vision.transformer.layers"],
        "vision_projector": ["transformer.vision.conv", "transformer.vision.linear_proj"],
        "llm": ["transformer.encoder"]
    },
}


MODEL_HF_PATH = OrderedDict()

MODEL_FAMILIES = OrderedDict()


def register_model(model_id: str, model_family_id: str, model_hf_path: str) -> None:
    if model_id in MODEL_HF_PATH or model_id in MODEL_FAMILIES:
        raise ValueError(f"Duplicate model_id: {model_id}")
    MODEL_HF_PATH[model_id] = model_hf_path
    MODEL_FAMILIES[model_id] = model_family_id


#=============================================================
# llava-1.5 --------------------------------------------------
register_model(
    model_id="llava-1.5-7b",
    model_family_id="llava-1.5",
    model_hf_path="llava-hf/llava-1.5-7b-hf"
)

register_model(
    model_id="llava-1.5-13b",
    model_family_id="llava-1.5",
    model_hf_path="llava-hf/llava-1.5-13b-hf"
)

# llava-1.6/next ---------------------------------------------
register_model(
    model_id="llava-1.6-vicuna-7b",
    model_family_id="llava-1.6",
    model_hf_path="llava-hf/llava-v1.6-vicuna-7b-hf"
)

register_model(
    model_id="llava-1.6-vicuna-13b",
    model_family_id="llava-1.6",
    model_hf_path="llava-hf/llava-v1.6-vicuna-13b-hf"
)

# llava-next-video -------------------------------------------
register_model(
    model_id="llava-next-video-7b",
    model_family_id="llava-next-video",
    model_hf_path="llava-hf/LLaVA-NeXT-Video-7B-hf"
)

register_model(
    model_id="llava-next-video-7b-32k",
    model_family_id="llava-next-video",
    model_hf_path="llava-hf/LLaVA-NeXT-Video-7B-32K-hf"
)

register_model(
    model_id="llava-next-video-34b",
    model_family_id="llava-next-video",
    model_hf_path="llava-hf/LLaVA-NeXT-Video-34B-hf"
)

# llava-interleave -------------------------------------------
register_model(
    model_id="llava-interleave-qwen-0.5b",
    model_family_id="llava-interleave",
    model_hf_path="llava-hf/llava-interleave-qwen-0.5b-hf"
)

register_model(
    model_id="llava-interleave-qwen-7b",
    model_family_id="llava-interleave",
    model_hf_path="llava-hf/llava-interleave-qwen-7b-hf"
)

# llava-onevision -------------------------------------------
register_model(
    model_id="llava-onevision-0.5b-ov",
    model_family_id="llava-onevision",
    model_hf_path="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
)

register_model(
    model_id="llava-onevision-7b-ov",
    model_family_id="llava-onevision",
    model_hf_path="llava-hf/llava-onevision-qwen2-7b-ov-hf"
)

register_model(
    model_id="llava-onevision-72b-ov",
    model_family_id="llava-onevision",
    model_hf_path="llava-hf/llava-onevision-qwen2-72b-ov-hf"
)

# qwen-vl ----------------------------------------------------
register_model(
    model_id="qwen-vl-chat",
    model_family_id="qwen-vl",
    model_hf_path="Qwen/Qwen-VL-Chat"
)

# phi3-v -----------------------------------------------------
register_model(
    model_id="phi3-v",
    model_family_id="phi3-v",
    model_hf_path="microsoft/Phi-3-vision-128k-instruct"
)

# qwen2-vl ---------------------------------------------------
register_model(
    model_id="qwen2-vl-2b-instruct",
    model_family_id="qwen2-vl",
    model_hf_path="Qwen/Qwen2-VL-2B-Instruct"
)

register_model(
    model_id="qwen2-vl-7b-instruct",
    model_family_id="qwen2-vl",
    model_hf_path="Qwen/Qwen2-VL-7B-Instruct"
)

register_model(
    model_id="qwen2-vl-72b-instruct",
    model_family_id="qwen2-vl",
    model_hf_path="Qwen/Qwen2-VL-72B-Instruct"
)

# qwen2.5-vl -------------------------------------------------
register_model(
    model_id="qwen2.5-vl-3b-instruct",
    model_family_id="qwen2.5-vl",
    model_hf_path="Qwen/Qwen2.5-VL-3B-Instruct"
)

register_model(
    model_id="qwen2.5-vl-7b-instruct",
    model_family_id="qwen2.5-vl",
    model_hf_path="Qwen/Qwen2.5-VL-7B-Instruct"
)

register_model(
    model_id="qwen2.5-vl-72b-instruct",
    model_family_id="qwen2.5-vl",
    model_hf_path="Qwen/Qwen2.5-VL-72B-Instruct"
)

# qwen3-vl ---------------------------------------------------
register_model(
    model_id="qwen3-vl-2b-instruct",
    model_family_id="qwen3-vl",
    model_hf_path="Qwen/Qwen3-VL-2B-Instruct"
)

register_model(
    model_id="qwen3-vl-4b-instruct",
    model_family_id="qwen3-vl",
    model_hf_path="Qwen/Qwen3-VL-4B-Instruct"
)

register_model(
    model_id="qwen3-vl-8b-instruct",
    model_family_id="qwen3-vl",
    model_hf_path="Qwen/Qwen3-VL-8B-Instruct"
)

register_model(
    model_id="qwen3-vl-32b-instruct",
    model_family_id="qwen3-vl",
    model_hf_path="Qwen/Qwen3-VL-32B-Instruct"
)

# llama-3.2-vision -------------------------------------------
register_model(
    model_id="llama-3.2-11b-vision-instruct",
    model_family_id="llama-3.2-vision",
    model_hf_path="meta-llama/Llama-3.2-11B-Vision-Instruct"
)

register_model(
    model_id="llama-3.2-90b-vision-instruct",
    model_family_id="llama-3.2-vision",
    model_hf_path="meta-llama/Llama-3.2-90B-Vision-Instruct"
)

# internvl2 --------------------------------------------------
register_model(
    model_id="internvl2-2b",
    model_family_id="internvl2",
    model_hf_path="OpenGVLab/InternVL2-2B"
)

register_model(
    model_id="internvl2-8b",
    model_family_id="internvl2",
    model_hf_path="OpenGVLab/InternVL2-8B"
)

register_model(
    model_id="internvl2-26b",
    model_family_id="internvl2",
    model_hf_path="OpenGVLab/InternVL2-26B"
)

# internvl2.5 ------------------------------------------------
register_model(
    model_id="internvl2.5-1b",
    model_family_id="internvl2.5",
    model_hf_path="OpenGVLab/InternVL2_5-1B"
)

register_model(
    model_id="internvl2.5-2b",
    model_family_id="internvl2.5",
    model_hf_path="OpenGVLab/InternVL2_5-2B"
)

register_model(
    model_id="internvl2.5-4b",
    model_family_id="internvl2.5",
    model_hf_path="OpenGVLab/InternVL2_5-4B"
)

register_model(
    model_id="internvl2.5-8b",
    model_family_id="internvl2.5",
    model_hf_path="OpenGVLab/InternVL2_5-8B"
)

register_model(
    model_id="internvl2.5-26b",
    model_family_id="internvl2.5",
    model_hf_path="OpenGVLab/InternVL2_5-26B"
)

register_model(
    model_id="internvl2.5-38b",
    model_family_id="internvl2.5",
    model_hf_path="OpenGVLab/InternVL2_5-38B"
)

register_model(
    model_id="internvl2.5-78b",
    model_family_id="internvl2.5",
    model_hf_path="OpenGVLab/InternVL2_5-78B"
)

# idefics2 ---------------------------------------------------
register_model(
    model_id="idefics2-8b",
    model_family_id="idefics2",
    model_hf_path="HuggingFaceM4/idefics2-8b"
)

# idefics3 ---------------------------------------------------
register_model(
    model_id="idefics3-8b-llama3",
    model_family_id="idefics3",
    model_hf_path="HuggingFaceM4/Idefics3-8B-Llama3"
)

# glm-4v -----------------------------------------------------
register_model(
    model_id="glm-4v-9b",
    model_family_id="glm-4v",
    model_hf_path="zai-org/glm-4v-9b"
)

#=============================================================


# sanity check
for model_family_id in MODEL_FAMILIES.values():
    assert model_family_id in COLLATORS, f"Collator not found for model family: {model_family_id}"
    assert model_family_id in LOADERS, f"Loader not found for model family: {model_family_id}"
    assert model_family_id in MODULE_KEYWORDS, f"Module keywords not found for model family: {model_family_id}"
    assert model_family_id in TO_LOAD_IMAGE, f"Image loading specification not found for model family: {model_family_id}"


if __name__ == "__main__":
    temp = "Model ID"
    ljust = 30
    print("Supported models:")
    print(f"  {temp.ljust(ljust)}: HuggingFace Path")
    print("  ------------------------------------------------")
    for model_id, model_hf_path in MODEL_HF_PATH.items():
        print(f"  {model_id.ljust(ljust)}: {model_hf_path}")
