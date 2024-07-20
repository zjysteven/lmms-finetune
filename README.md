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

0. See if the model you want to finetune is supported by running
```bash
python supported_models.py
```
which will show things like
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

1. 


### Training


### Known limitations

- memory cost especially for full finetuning (due to hf implementation)
- freezing the vision modules for now
- does not have good support for training on mix of multimodal and unimodal data


## Acknowledgements

The codebase borrows from, is inspired by, or builds upon the following code, repos, and/or libraries: [LLaVA](https://github.com/haotian-liu/LLaVA), [Qwen](https://github.com/QwenLM/Qwen-VL/blob/master/finetune.py), [transformers](https://github.com/huggingface/transformers), a [sample finetuning script](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LLaVa/Fine_tune_LLaVa_on_a_custom_dataset_(with_PyTorch_Lightning).ipynb) using Lightning by huggingface staff, etc.