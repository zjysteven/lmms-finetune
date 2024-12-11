import argparse

import torch
import transformers
from peft import PeftModel

from loaders import LOADERS
from supported_models import MODEL_HF_PATH, MODEL_FAMILIES


def merge_lora(args):
    model_hf_path = MODEL_HF_PATH[args.model_id]
    model_local_path = model_local_path if args.model_local_path else model_hf_path
    model_family_id = MODEL_FAMILIES[args.model_id]
    
    bnb_config = None
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        print("Quantization for LLM enabled...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", 
        )
    
    loader = LOADERS[model_family_id](
        model_hf_path=model_hf_path,
        model_local_path=model_local_path,
        compute_dtype=torch.bfloat16,
        bnb_config=bnb_config,
        device_map="cpu",
    )
    model, tokenizer, processor, config = loader.load()
    
    print("Loading LoRA weights...")
    model = PeftModel.from_pretrained(model, args.model_path)
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    print("Model is loaded...")

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)
    processor.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--model_local_path", type=str, default="")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--load_4bit", type=bool, default=False)
    parser.add_argument("--save_model_path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args)