import re
from typing import Dict, List, Sequence, Union

import numpy as np
import PIL
import torch
from transformers.image_utils import get_image_size, to_numpy_array
from transformers.utils import logging

from . import register_collator
from .base import BaseDataCollator


logger = logging.get_logger(__name__)


@register_collator("llava-1.5")
class LLaVA15DataCollator(BaseDataCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # some parsing
        # the dataset implementation assume conversations are [user, assistant, user, assistant, ...]
        images: List[List[PIL.Image.Image]] = [instance["images"] for instance in instances]
        system_prompts: List[Union[str, None]] = [instance["system_prompt"] for instance in instances]
        conversations: List[List] = [instance["conversations"] for instance in instances]
        max_len = self.tokenizer.model_max_length

        # constants
        image_token_id = self.config.image_token_index
        patch_size = self.processor.patch_size
        vision_feature_select_strategy = self.processor.vision_feature_select_strategy

        input_ids = []
        labels = []
        all_vision_inputs = []
        
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
            
            assert len(cur_images) == cur_num_images, "Number of image tokens does not match the number of images"

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

            # preprocess image
            vision_inputs = self.processor.image_processor(cur_images, return_tensors="pt")
            all_vision_inputs.append(vision_inputs)

            if vision_inputs.get("pixel_values") is not None:
                if patch_size is not None and vision_feature_select_strategy is not None:
                    # Replace the image token with the expanded image token sequence
                    pixel_values = vision_inputs["pixel_values"]
                    height, width = get_image_size(to_numpy_array(pixel_values[0]))
                    num_image_tokens = (height // patch_size) * (width // patch_size) + 1
                    if vision_feature_select_strategy == "default":
                        num_image_tokens -= 1

                    repeat = torch.where(cur_input_ids == image_token_id, num_image_tokens, 1).squeeze()
                    cur_input_ids = cur_input_ids.repeat_interleave(repeat, dim=1)
                    cur_assistant_masks = cur_assistant_masks.repeat_interleave(repeat, dim=1)
                else:
                    logger.warning_once(
                        "Expanding inputs for image tokens in LLaVa should be done in processing. "
                        "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
                        "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
                        "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
                    )

            # manual truncation
            if cur_input_ids.shape[1] > max_len:
                cur_input_ids = cur_input_ids[:, :max_len]
                cur_assistant_masks = cur_assistant_masks[:, :max_len]
            cur_labels = cur_input_ids.clone()

            # mask question tokens
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

        vision_inputs = {}
        for key in all_vision_inputs[0].keys():
            vision_inputs[key] = torch.cat([x[key] for x in all_vision_inputs])

        return dict(
            **vision_inputs,
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.PAD_TOKEN_ID),
        )