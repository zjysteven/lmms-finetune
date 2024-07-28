from collections import OrderedDict

from collators import COLLATORS
from datasets import TO_LOAD_IMAGE
from loaders import LOADERS


MULTIMODAL_KEYWORDS = {
    "llava-1.5": ["multi_modal_projector", "vision_tower"],
    "llava-interleave": ["multi_modal_projector", "vision_tower"],
    "llava-next-video": ["multi_modal_projector", "vision_tower", "image_newline", "vision_resampler"],
    "qwen-vl": ["transformer.visual"],
    "phi3-v": ["vision_embed_tokens"],
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

# llava-1.6 --------------------------------------------------
# register_model(
#     model_id="llava-1.6-mistral-7b",
#     model_family_id="llava-1.6",
#     model_hf_path="llava-hf/llava-v1.6-mistral-7b-hf"
# )

# register_model(
#     model_id="llava-1.6-vicuna-7b",
#     model_family_id="llava-1.6",
#     model_hf_path="llava-hf/llava-v1.6-vicuna-7b-hf"
# )

# register_model(
#     model_id="llava-1.6-vicuna-13b",
#     model_family_id="llava-1.6",
#     model_hf_path="llava-hf/llava-v1.6-vicuna-13b-hf"
# )

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

#=============================================================


# sanity check
for model_family_id in MODEL_FAMILIES.values():
    assert model_family_id in COLLATORS, f"Collator not found for model family: {model_family_id}"
    assert model_family_id in LOADERS, f"Loader not found for model family: {model_family_id}"
    assert model_family_id in MULTIMODAL_KEYWORDS, f"Multimodal keywords not found for model family: {model_family_id}"
    assert model_family_id in TO_LOAD_IMAGE, f"Image loading specification not found for model family: {model_family_id}"


if __name__ == "__main__":
    temp = "Model ID"
    ljust = 30
    print("Supported models:")
    print(f"  {temp.ljust(ljust)}: HuggingFace Path")
    print("  ------------------------------------------------")
    for model_id, model_hf_path in MODEL_HF_PATH.items():
        print(f"  {model_id.ljust(ljust)}: {model_hf_path}")
