# Inference

ℹ️ **The rule of thumb is that lmms-finetune is only related to the finetuning and is decoupled from the inference. Once you load your finetuned model, all the other components/operations are carried out in the exact same way as how you would inference with the pretrained model, which is always made clear by the corresponding huggingface model card.**

Taking LLaVA-1.5 as an example, this is how you load the model and processor according to the huggingface model card:
```python
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
)

processor = AutoProcessor.from_pretrained(model_id)
```

----------------

- After full finetuning or lora finetuning, you can load the model and processor like this:
```python
from transformers import AutoProcessor, LlavaForConditionalGeneration

original_model_id = "llava-hf/llava-1.5-7b-hf"
model_id = "path/to/your/model"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)
# processor is not changed so we still load from the original model repo
processor = AutoProcessor.from_pretrained(original_model_id)
```


- After q-lora finetuning, you can load the model and processor like this:
```python
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4", 
)

original_model_id = "llava-hf/llava-1.5-7b-hf"
model_id = "path/to/your/model"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True, 
)
# processor is not changed so we still load from the original model repo
processor = AutoProcessor.from_pretrained(original_model_id)
```


# Merge Model

You can further use the `merge_lora_weights.py` to merge the finetuned weights with the base model to get a standalone model. See below examples.


- After lora finetuning
```bash
python merge_lora_weights.py \
    --model_id model_id \
    --model_local_path /path/to/local/model \
    --model_path /path/to/saved/model \
    --model_save_path /path/to/output \
    --load_model
```

----------------

- After q-lora finetuning
```bash
python merge_lora_weights.py \
    --model_id model_id \
    --model_local_path /path/to/local/model \
    --model_path /path/to/saved/model \
    --model_save_path /path/to/output \
    --load_model \
    --load_4bit
```

----------------

- After full finetuning
```bash
python merge_lora_weights.py \
    --model_id model_id \
    --model_local_path /path/to/local/model \
    --model_path /path/to/saved/model
```