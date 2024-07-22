# Adding a new model

Glad that you want to use this codebase for finetuning a new model. Below I will show you, step by step, how to add a new model. This in fact is exactly how I added the current supported models. We will start from the easiest part, and then move on to the more complex parts. Trust me it's not that hard.


## Step 1: Register the model in `supported_models.py`
```python
register_model(
    model_id="llava-1.5-7b",
    model_family_id="llava-1.5",
    model_hf_path="llava-hf/llava-1.5-7b-hf"
)
```
Taking the registry of LLaVA-1.5 as an example, you can see that all we need to do is to register the `model_id` (a unique identifier for the model), the `model_family_id` (to identify the family of models that share the same architecture), and the `model_hf_path` (the path to the model in the Hugging Face Hub).


Then at the top of `supported_models.py`, you will see a dictionary `MULTIMODAL_KEYWORDS`, which is a mapping from the model family id to a list of vision module keywords. This is used freeze the vision modules during finetuning. To find the vision module keywords, you just need to take a look at the huggingface's model implementation (the init function of the model in `modeling_xxxx.py`). Again taking LLaVA-1.5 as an example, according to [here](https://github.com/huggingface/transformers/blob/0fdea8607d7e01eb0e38a1ebeb7feee30a22f0cf/src/transformers/models/llava/modeling_llava.py#L237-L247) it can be seen that the vision module keywords are `["vision_tower", "multi_modal_projector"]`.


## Step 2: Implement the loader for the model
The loader helps to load the model, processor, and tokenizer from huggingface. It basically just calls `.from_pretrained()`. You can find plenty of examples by browsing through the existing loaders under the `loaders` directory. The actual loading operations would depend on the specific model, but you can always refer to the model card on huggingface which typically shows how to load the model/processor/tokenizer.


## Step 3: Add `TO_LOAD_IMAGE` key-value pair in `datasets.py`
Some models (e.g., Qwen-VL-Chat) encode the image in a very specific way, where the path to the image is encoded in the prompt, and the image won't be loaded until the tokenization. In this case the returned entry from the dataset doesn't have to load the image. This is why we have the `TO_LOAD_IMAGE` dictionary in `datasets.py`. You can see that for Qwen-VL-Chat (whose model family is `qwen-vl`), `TO_LOAD_IMAGE["qwen-vl"]` is set to `False`. If the model you are adding doesn't have this special requirement, you can simply set it to `True`.


## Step 4: Implement the collator for the model
Now we have proceeded to the relatively most complex part. The collator prepares the actual inputs (e.g., `input_ids`, `pixel_values`, `attention_masks`, etc.) for the model and is the place where tokenization, image/video preprocessing happens. That is also where we address different chat templates of different models. Again, there are plenty of examples in the `collators` directory, which you can take a look to get a sense. The actual implementation would be model-specific, but again you can always refer to the model card and the chat template on huggingface to see how the inputs are prepared (e.g., what is the image token, what is the user/assistant token, etc.).

Note that if you find this part too complex, you can choose another simpler strategy which leverages the `apply_chat_template` function provided by huggingface, as showcased by [this script](https://github.com/huggingface/trl/blob/main/examples/scripts/vsft_llava.py). In this way the prompt is automatically formed, and you don't have to worry about anything. However, one issue with such strategy is that the constructed `labels` will be exactly the same as your whole `input_ids` (see [this specific line of code](https://github.com/huggingface/trl/blob/4e85bd75a9dfca0074eef3a90130054c283eed39/examples/scripts/vsft_llava.py#L169)), meaning that the model will be learning from both the user's questions and the assistant's responses. This could be a problem when the average length of responses is much shorter than the questions. This is why in this codebase we choose to manually deal with the prompt construction in the collator such that the `labels` are only the assistant's responses.
