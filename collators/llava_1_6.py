import re
from typing import Dict, List, Sequence, Union

import numpy as np
import PIL
import torch
from transformers.image_utils import get_image_size, to_numpy_array
from transformers.models.llava_next.processing_llava_next import LlavaNextProcessorKwargs
from transformers.utils import logging

from . import register_collator
from .base import BaseDataCollator


logger = logging.get_logger(__name__)


@register_collator("llava-1.6")
class LLaVA16DataCollator(BaseDataCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        output_kwargs = self.processor._merge_kwargs(
            LlavaNextProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
        )

        vision_inputs = dict()
        images: List[List[PIL.Image.Image]] = [x for instance in instances for x in instance["images"]]
        if len(images) > 0:
            vision_inputs.update(**self.processor.image_processor(images, return_tensors="pt", **output_kwargs["images_kwargs"]))

        # some parsing
        images: List[List[PIL.Image.Image]] = [instance["images"] for instance in instances]
        system_prompts: List[Union[str, None]] = [instance["system_prompt"] for instance in instances]
        conversations: List[List] = [instance["conversations"] for instance in instances]

        # constants
        max_len = self.tokenizer.model_max_length
        image_token_id = self.config.image_token_index
        patch_size = self.processor.patch_size
        vision_feature_select_strategy = self.processor.vision_feature_select_strategy

        input_ids = []
        labels = []
        
        for system_prompt, cur_images, cur_convs in zip(system_prompts, images, conversations):
            cur_num_images = 0
            cur_input_ids = []
            cur_labels = []
            
            cur_text = []
            if system_prompt is not None:
                cur_text.append({
                    "role": "system",
                    "content": [{"text": system_prompt}]
                })
            
            for i, text in enumerate(cur_convs):
                if i % 2 == 0:
                    num_images = len([m.start() for m in re.finditer("<image>", text)])
                    cur_num_images += num_images

                    # .strip(): whitespaces and newlines are handled by chat_template
                    text = text.replace("<image>", "").strip()

                    cur_text.append({
                        "role": "user",
                        "content": [{"type": "text", "text": text}] + \
                            [{"type": "image"}] * num_images
                    })
                else:
                    cur_text.append({
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": text},
                        ]
                    })

            assert len(cur_images) == cur_num_images, "Number of images does not match the number of image tokens"
            
            temp = self.processor.apply_chat_template(
                cur_text,
                add_generation_prompt=False,
                tokenize=True,
                return_assistant_tokens_mask=True,
                return_dict=True,
                return_tensors="pt",
                truncation=False # the assistant tokens mask seems wrong when truncation is enabled
            )
            cur_input_ids = temp["input_ids"]
            cur_assistant_masks = torch.tensor(temp["assistant_masks"], dtype=torch.bool).unsqueeze(0)

            # expand image tokens
            temp_vision_inputs = self.processor.image_processor(cur_images, do_pad=True, return_tensors="pt")
            if temp_vision_inputs.get("pixel_values") is not None:
                if patch_size is not None and vision_feature_select_strategy is not None:
                    image_sizes = temp_vision_inputs["image_sizes"]
                    height, width = get_image_size(to_numpy_array(temp_vision_inputs["pixel_values"][0][0]))

                    num_image_tokens_list = []
                    for image_size in image_sizes:
                        orig_height, orig_width = image_size
                        num_image_tokens = self.processor._get_number_of_features(orig_height, orig_width, height, width)
                        if vision_feature_select_strategy == "default":
                            num_image_tokens -= 1
                        num_image_tokens_list.append(num_image_tokens)
                    
                    repeat = torch.ones(cur_input_ids.shape[1], dtype=torch.long)
                    repeat[torch.where(cur_input_ids == image_token_id)[1]] = torch.tensor(num_image_tokens_list, dtype=torch.long)
                    cur_input_ids = cur_input_ids.repeat_interleave(repeat, dim=1)
                    cur_assistant_masks = cur_assistant_masks.repeat_interleave(repeat, dim=1)
                else:
                    logger.warning_once(
                        "Expanding inputs for image tokens in LLaVa-NeXT should be done in processing. "
                        "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
                        "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
                        "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
                    )

            # manual truncation
            if cur_input_ids.shape[1] > max_len:
                cur_input_ids = cur_input_ids[:, :max_len]
                cur_assistant_masks = cur_assistant_masks[:, :max_len]
            cur_labels = cur_input_ids.clone()

            if self.mask_question_tokens:
                assert cur_labels.shape == cur_assistant_masks.shape, "Label and mask shapes do not match"
                cur_labels = torch.where(cur_assistant_masks, cur_labels, self.IGNORE_TOKEN_ID)
            
            assert cur_input_ids.shape == cur_labels.shape, "Input and label shapes do not match"

            # padding
            if cur_input_ids.shape[1] < max_len:
                cur_input_ids = torch.cat([
                    cur_input_ids,
                    torch.full(
                        (cur_input_ids.shape[0], max_len - cur_input_ids.shape[1]),
                        self.PAD_TOKEN_ID,
                        dtype=cur_input_ids.dtype,
                        device=cur_input_ids.device
                    )
                ], dim=1)
                cur_labels = torch.cat([
                    cur_labels,
                    torch.full(
                        (cur_labels.shape[0], max_len - cur_labels.shape[1]),
                        self.IGNORE_TOKEN_ID,
                        dtype=cur_labels.dtype,
                        device=cur_labels.device
                    )
                ], dim=1)

            input_ids.append(cur_input_ids)
            labels.append(cur_labels)

        input_ids = torch.cat(input_ids)
        labels = torch.cat(labels)

        return dict(
            **vision_inputs,
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.PAD_TOKEN_ID),
        )