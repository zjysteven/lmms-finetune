# Enabling the finetuning of the latest Large Multimodal Models

| :exclamation: This codebase will NOT be under active maintainence/update after November 2024, as the main contributor/maintainer, [Jingyang Zhang](https://zjysteven.github.io/) will be graduating.|
|-----------------------------------------|


## About

More and more large multimodal models (LMMs) are being released from time to time, but the finetuning of these models is not always straightforward. This codebase aims to provide a unified, minimal structure for LMM finetuning. Key design ideas include:
- the components of the finetuning process (e.g., model loading, data collating) are abstracted, allowing one to easily integrate the latest LMMs into this codebase and finetune them with minimal effort;
- for all LMMs the 🤗huggingface's official implementation is used, so that after finetuning one can do inference and everything else in the exact same way as earlier with the HF model;
- the codebase is kept as simple/lightweight as possible, so that it is easy to understand and modify.


The codebase is quite flexible. It supports the finetuning of various types of LMMs, including:
- :city_sunrise: single image models: [LLaVA-1.5](https://huggingface.co/collections/llava-hf/llava-15-65f762d5b6941db5c2ba07e0), [LLaVA-1.6/NeXT](https://huggingface.co/collections/llava-hf/llava-next-65f75c4afac77fd37dbbe6cf), [Phi-3-Vision](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct), [Llama-3.2-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)
- :bookmark_tabs: multiple/interleaved image models: [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat), [Qwen2-VL-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct),  [LLaVA-NeXT-Interleave](https://huggingface.co/collections/llava-hf/llava-interleave-668e19a97da0036aad4a2f19)
- :movie_camera: video models: [LLaVA-NeXT-Video](https://huggingface.co/collections/llava-hf/llava-next-video-6666a9173a64c7052930f153)
- :rocket: unified models: [LLaVA-Onevision](https://huggingface.co/collections/llava-hf/llava-onevision-66bb1e9ce8856e210a7ed1fe)

See [supported_models.md](docs/supported_models.md) for the full list of supported models. For training strategy, 1) full-finetuning, 2) lora, and 3) q-lora are supported for the LLM component, while 1) full-finetuning and 2) lora are supported for the vision encoder/backbone.

<!---
*TODOS:* 
- [x] Support training with text-only data.
- [x] Support tuning vision models and projectors.
- [ ] Add more models, including llava-onvision, idefics2, glm4-v, minicpm, etc.


:raising_hand: If you would like to have a model available, feel free to open an issue.
-->

<details>
<summary>What's different from other training frameworks, e.g., LLaMA-Factory, xtuner, swift?</summary>

These are great projects/frameworks with large scale and high-degree optimization. However, due to their scale and complexity, they could be less transparent and less easy to get started (e.g., I personally feel quite lost when trying to use those frameworks, with a bunch of questions like "how should I format my data"). This codebase (lmms-finetune) is instead designed to be lightweight and simple, meaning that it's much more likely for you to quickly get started and be able to know almost every detail of the training process if you want. In other words, this is a minimal workable codebase that supports LMM finetuning, while facilitating quick experiments, flexible modifications, and easy integrations of new models.
</details>

## News

- **2024/12/16**: Thanks to the contribution from [lavinal712 (Yuqian)](https://github.com/lavinal712), training with Llama-3.2-Vision is now supported. Also there is a useful script `merge_lora_weights.py` added.
- **2024/10/16**: We added LLaVA-Onevision. See a caveat when using LLaVA-Onevision [here](https://github.com/zjysteven/lmms-finetune/issues/43). Also we updated the collators to stay in line with the new processing of LLaVA models in transformers.
- **2024/08/28**: Finetuning with gradio webui interface is supported. Try `python webui.py`.
- **2024/07/30**: Finetuning of vision encoder and projector is now supported.
- **2024/07/25**: Several things are improved. We have *1)* released a [colab notebook](https://colab.research.google.com/drive/139XypY8_wdLgyLXYE_Zve7Hjd809fVpK?usp=sharing) demonstrating a full, successful training run with LLaVA-NeXT-Video-7B (happy to hear from people that they succeeded in [their cases](https://github.com/zjysteven/lmms-finetune/issues/7#issuecomment-2249864887) too); *2)* supported having text-only samples in the training set (see [this](docs/dataset.md) for one note).
- **2024/07/20**: Initial release of the codebase. More models and optimizations are coming soon. Stay tuned!


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

A workable example training run (of LLaVA-NeXT-Video-7B) is showcased in this [colab notebook](https://colab.research.google.com/drive/139XypY8_wdLgyLXYE_Zve7Hjd809fVpK?usp=sharing), which is a good starting point to get a sense of how to use this codebase. The following sections provide a more detailed guide on how to finetune a model.

<details>
<summary><b>0. See if the model you want to finetune is supported</b></summary>

Browse [supported_models.md](docs/supported_models.md). Or run `python supported_models.py`, which will for example show things like
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
  llava-onevision-0.5b-ov       : llava-hf/llava-onevision-qwen2-0.5b-ov-hf
  llava-onevision-7b-ov         : llava-hf/llava-onevision-qwen2-7b-ov-hf
  llava-onevision-72b-ov        : llava-hf/llava-onevision-qwen2-72b-ov-hf
  qwen-vl-chat                  : Qwen/Qwen-VL-Chat
  phi3-v                        : microsoft/Phi-3-vision-128k-instruct
  qwen2-vl-2b-instruct          : Qwen/Qwen2-VL-2B-Instruct
  qwen2-vl-7b-instruct          : Qwen/Qwen2-VL-7B-Instruct
  llama-3.2-11b-vision-instruct : meta-llama/Llama-3.2-11B-Vision-Instruct
  llama-3.2-90b-vision-instruct : meta-llama/Llama-3.2-90B-Vision-Instruct
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
        "conversations": [
            {
                "from": "human",
                "value": "<video>What is this video about?"
            },
            {
                "from": "gpt",
                "value": "This video shows a baby crying."
            },
        ]
    }
]
```
The image and video token is assumed to be `<image>` and `<video>`. We adopt this format for its readability. Our dataset implementation is general enough to support variations within this format, e.g., multiple image/video inputs in a sample, text-only sample etc. For more details, see the [dataset documentation](docs/dataset.md) and find how flexible this json file can be. There are also mutiple example json files under [example_data](./example_data) for reference.

Besides this json file, the actual videos and images are by default assumed to be stored in their corresponding folders, and then the paths in the json file should be relative to the video/image root folder. Or the paths can simply be absolute paths.

:warning: **If you have text-only entries in your training dataset:** the training is likely to fail at some point if 1) your `per_device_batch_size` is 1, or 2) the number of text-only instances dominate the number of multi-modal instances. This is due to a limitation/bug of deepspeed. If neither of the above two conditions is met, no worries, we got you covered.
</details>


<details>
<summary><b>2. Perform finetuning</b></summary>

Modify the sample training bash script, [example_video.sh](./example_scripts/example_video.sh) or [example_image.sh](example_image.sh) (there are no differences other than different model ID and dataset filepath), to specify arguments including the target model, data path, etc. There are comments that explain each argument's meaning. Then simply kick off the training by running the bash script `bash example_scripts/example_video.sh` or `bash example_scripts/example_image.sh`. Note that to exactly run the provided [example_video.sh](./example_scripts/example_video.sh), you will need to download the video clips from ShareGPT4Video; see [here](example_data/videos/ego4d/README.md) for instructions.

:chart_with_upwards_trend:*If you prefer graphical interface*, simply run `python webui.py` to lauch the gradio interface for finetuning.
</details>


<details>
<summary><b>3. Inference with finetuned model</b></summary>

The key here is to correctly load the finetuned model, after that everything is the same as how you would do inference with the corresponding model from huggingface. Refer to the [inference documentation](docs/inference.md) for more details, including how to use `merge_lora_weights.py` to easily obtain a standalone model. Again you can refer to [this colab](https://colab.research.google.com/drive/139XypY8_wdLgyLXYE_Zve7Hjd809fVpK?usp=sharing) for a complete example.
</details>


## Acknowledgements

We want to thank the huggingface team for actively integrating newest models in the transformers library. Also, the example finetuning scripts (e.g., [this](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LLaVa/Fine_tune_LLaVa_on_a_custom_dataset_(with_PyTorch_Lightning).ipynb), [this](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LLaVa-NeXT/Fine_tune_LLaVaNeXT_on_a_custom_dataset_(with_PyTorch_Lightning).ipynb), and [this](https://colab.research.google.com/drive/1dTdro-k7NFqRgGq5-TlGHM-6k2sYQhXp#scrollTo=4ccbd183-f15a-4f94-a526-9ceeec3f61e0)) made by HF staff, [Niels Rogge](https://github.com/NielsRogge) and [Raushan Turganbay](https://github.com/zucchini-nlp), are very helpful and lay the foundation for this codebase. We also especially thank [Raushan Turganbay](https://github.com/zucchini-nlp) for her generous discussions and feedbacks on this project.

The codebase borrows from, is inspired by, or builds upon the following code, repos, and/or libraries: [LLaVA](https://github.com/haotian-liu/LLaVA), [Qwen](https://github.com/QwenLM/Qwen-VL/blob/master/finetune.py), [transformers](https://github.com/huggingface/transformers), etc.

## Citation
If you use lmms-finetune in your research/project, we'd be very happy if you could 1) give us a star, 2) share this repo with others, or 3) cite this codebase:
```
@software{Zhang_lmms-finetune,
author = {Zhang, Jingyang and Lin, Yueqian},
license = {Apache-2.0},
title = {{lmms-finetune}},
url = {https://github.com/zjysteven/lmms-finetune}
}
```
