import re
from typing import Dict, List, Sequence, Union

import numpy as np
import PIL
import torch
from transformers.image_utils import get_image_size, to_numpy_array
from transformers.models.llava.processing_llava import LlavaProcessorKwargs
from transformers.utils import logging

from . import register_collator
from .base import BaseDataCollator
from .chat_template_monkey_patch import apply_chat_template


logger = logging.get_logger(__name__)


@register_collator("glm-4v")
class GLM4VDataCollator(BaseDataCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # some parsing
        images: List[List[PIL.Image.Image]] = [instance["images"] for instance in instances]
        system_prompts: List[Union[str, None]] = [instance["system_prompt"] for instance in instances]
        conversations: List[List] = [instance["conversations"] for instance in instances]

        # constants
        max_len = self.tokenizer.model_max_length
        empty_image = PIL.Image.new("L", (224, 224), 0).convert("RGB")

        input_ids = []
        labels = []
        images_tensor = []

        for system_prompt, cur_images, cur_convs in zip(system_prompts, images, conversations):
            cur_num_images = 0
            cur_input_ids = []
            cur_labels = []
            cur_assistant_masks = []

            cur_text = []
            if system_prompt is not None:
                cur_text.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                })

            for i, text in enumerate(cur_convs):
                if i % 2 == 0:
                    num_images = len([m.start() for m in re.finditer("<image>", text)])
                    cur_num_images += num_images
                    assert num_images <= 1, "GLM-4V currently only supports single-image input"

                    # .strip(): whitespaces and newlines are handled by chat_template
                    # text = text.replace("<image>", "").strip()

                    cur_text.append({
                        "role": "user",
                        "image": cur_images[cur_num_images - 1] if num_images == 1 else empty_image,
                        "content": text
                    })
                else:
                    cur_text.append({
                        "role": "assistant",
                        "content": text,
                    })
            
            assert len(cur_images) == cur_num_images, "Number of image tokens does not match the number of images"

            temp = self.tokenizer.apply_chat_template(
                cur_text,
                padding=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                truncation=True,
            )
            cur_input_ids = temp["input_ids"][:, 2:]
            cur_labels = cur_input_ids.clone()
            images_tensor.append(temp["images"])

            # mask question tokens
            if self.mask_question_tokens:
                offset = 0
                cur_labels = torch.full_like(cur_input_ids, self.IGNORE_TOKEN_ID)

                for block in cur_text:
                    temp = self.tokenizer.apply_chat_template(
                        [block],
                        padding=False,
                        return_dict=True,
                        return_tensors="pt",
                        truncation=True,
                    )
                    length = len(temp["input_ids"][0])

                    if block["role"] == "assistant":
                        cur_labels[:, offset: offset + length] = cur_input_ids[:, offset: offset + length]

                    offset += length

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
        images_tensor = torch.cat(images_tensor)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.PAD_TOKEN_ID),
            images=images_tensor,
        )