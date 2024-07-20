# Enabling the finetuning of the latest Large Multimodal Models


## About

More and more large multimodal models (LMMs) are being released from time to time, but the finetuning of these models is not always straightforward. This codebase aims to provide a unified structure for LMM finetuning. Key design ideas include:
- the components of the finetuning process (e.g., model loading, data collating) are abstracted, allowing one to easily integrate the latest LMMs into this codebase and finetune them with minimal effort;
- for all LMMs the huggingface's official implementation is used, so that after finetuning one can do inference and everything else in the exact same way as the earlier;
- the codebase is kept as simple/lightweight as possible, so that it is easy to understand and modify.


The codebase is quite flexible. Despite being at an early stage, it already supports the finetuning of various types of LMMs, including:
- :city_sunrise: single image models: [LLaVA-1.5](https://huggingface.co/collections/llava-hf/llava-15-65f762d5b6941db5c2ba07e0)
- :bookmark_tabs: multiple/interleaved image models: [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat), [LLaVA-NeXT-Interleave](https://huggingface.co/collections/llava-hf/llava-interleave-668e19a97da0036aad4a2f19)
- :movie_camera: video models: [LLaVA-NeXT-Video](https://huggingface.co/collections/llava-hf/llava-next-video-6666a9173a64c7052930f153)

For training strategy, 1) full-finetuning, 2) lora, and 3) q-lora are supported.


## Installation

```bash
# clone this repo
git clone https://github.com/zjysteven/lmms-finetune.git

# set up a conda environment
conda create -n lmms-finetune python=3.10 -y
conda activate lmms-finetune
## this will install the latest version of torch
## feel free to change it to a specific version
python -m pip install -r requirements.txt

## optionally install flash attention
python -m pip install --no-cache-dir --no-build-isolation flash-attn
```

## Usage

<details>
<summary><b>0. See if the model you want to finetune is supported</b></summary>

Run `python supported_models.py`, which will show things like
```
Supported models:
  Model ID                      : HuggingFace Path
  ------------------------------------------------
  llava-1.5-7b                  : llava-hf/llava-1.5-7b-hf
  llava-1.5-13b                 : llava-hf/llava-1.5-13b-hf
  llava-next-video-7b           : llava-hf/LLaVA-NeXT-Video-7B-hf
  llava-next-video-7b-32k       : llava-hf/LLaVA-NeXT-Video-7B-32K-hf
  llava-next-video-34b          : llava-hf/LLaVA-NeXT-Video-34B-hf
  llava-interleave-qwen-0.5b    : llava-hf/llava-interleave-qwen-0.5b-hf
  llava-interleave-qwen-7b      : llava-hf/llava-interleave-qwen-7b-hf
  qwen-vl-chat                  : Qwen/Qwen-VL-Chat
```
:raised_hand: Don't see the one you want? Check out this [guide](docs/add_new_model.md) for step-by-step instructions on how to add a new model.
</details>


<details>
<summary><b>1. Prepare your finetuning data</b></summary>

Similar to LLaVA, we expect the data to be in a json file containing a list of dictionaries, where each dictionary is a sample.
```json
[
    {
        "system_prompt": "You are a helpful assistant.",
        "video": "path/to/video1.mp4",
        "num_frames": 10,
        "conversations": [
            {
                "from": "human",
                "value": "<video>\nWhat is this video about?"
            },
            {
                "from": "gpt",
                "value": "This video shows a baby crying."
            },
        ]
    }
]
```
The image and video token is assumed to be `<image>` and `<video>`. We adopt this format for its readability. Our dataset implementation is general enough to support variations within this format, e.g., multiple image/video inputs in a sample. For more details, see the [dataset documentation](docs/dataset.md) where we go over several examples to see how flexible this json file can be.

The actual videos and images can be stored in their corresponding folders, and then the paths in the json file should be relative to the video/image root folder. Or the paths can simply be absolute paths.
</details>


<details>
<summary><b>2. Perform finetuning</b></summary>

Modify the sample training bash script `example.sh` to specify arguments including the target model, data path, etc. Refer to the [training documentation](docs/training.md) for more details on the arguments and their meanings. Then simply kick off the training by running the bash script `bash example.sh`.
</details>


<details>
<summary><b>3. Inference with finetuned model</b></summary>

The key here is to correctly load the finetuned model, after that everything is the same as how you would do inference with the corresponding model from huggingface. Refer to the [inference documentation](docs/inference.md) for more details.
</details>


<details>
<summary>Known limitations</summary>

- :neutral_face: Due to huggingface's implementation (e.g., the vision encoder's hidden states are saved, see [this](https://github.com/huggingface/transformers/blob/0fdea8607d7e01eb0e38a1ebeb7feee30a22f0cf/src/transformers/models/llava/modeling_llava.py#L425)), the memory cost can be high especially for full finetuning.
- :neutral_face: Currently all vision modules are freezed for simplicity.
- :warning: Due to [an unsolved issue](https://github.com/microsoft/DeepSpeed/issues/3156) in deepspeed (all parameters have to be used in the forward pass), currently the training might not succeed if you have text-only data in your dataset.
</details>

## Acknowledgements

The codebase borrows from, is inspired by, or builds upon the following code, repos, and/or libraries: [LLaVA](https://github.com/haotian-liu/LLaVA), [Qwen](https://github.com/QwenLM/Qwen-VL/blob/master/finetune.py), [transformers](https://github.com/huggingface/transformers), a [sample finetuning script](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LLaVa/Fine_tune_LLaVa_on_a_custom_dataset_(with_PyTorch_Lightning).ipynb) using Lightning by huggingface staff, etc.