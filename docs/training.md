# Training Tutorial

placeholder


## Merge Model
- After lora finetuning, you can merge the model like this:
```bash
python merge_lora_weights.py \
    --model_id model_id \
    --model_local_path /path/to/local/model \
    --model_path /path/to/saved/model \
    --model_save_path /path/to/output \
    --load_model
```

----------------

- After q-lora finetuning, you can merge the model like this:
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

- After full finetuning, you can merge the model like this:
```bash
python merge_lora_weights.py \
    --model_id model_id \
    --model_local_path /path/to/local/model \
    --model_path /path/to/saved/model
```