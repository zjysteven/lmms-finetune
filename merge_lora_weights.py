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
    model_save_path = args.model_save_path if args.model_save_path else args.model_path
    
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
    model, tokenizer, processor, config = loader.load(args.load_model)
    print(args.load_model)
    
    if args.load_model:
        print("Loading LoRA weights...")
        model = PeftModel.from_pretrained(model, args.model_path)
        print("Merging LoRA weights...")
        model = model.merge_and_unload()
        print("Model is loaded...")
        model.save_pretrained(model_save_path)
    
    tokenizer.save_pretrained(model_save_path)
    processor.save_pretrained(model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--model_local_path", type=str, default="")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_save_path", type=str, default="")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")

    args = parser.parse_args()

    merge_lora(args)