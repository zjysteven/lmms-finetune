import re
from typing import Dict, List, Sequence, Union

import numpy as np
import PIL
import torch
from transformers.image_utils import get_image_size, to_numpy_array
from transformers.models.mllama.processing_mllama import MllamaProcessorKwargs
from transformers.utils import logging

from . import register_collator
from .base import BaseDataCollator
from .chat_template_monkey_patch import apply_chat_template


logger = logging.get_logger(__name__)


@register_collator("llama-3.2-vision")
class LLaMA32VisionDataCollator(BaseDataCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # monkey patch to include bos tokens
        self.tokenizer.apply_chat_template = apply_chat_template.__get__(self.tokenizer)

        output_kwargs = self.processor._merge_kwargs(
            MllamaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
        )
        
        vision_inputs = dict()
        images: List[List[PIL.Image.Image]] = [x for instance in instances for x in instance["images"]]
        if len(images) > 0:
            image_features = self.processor.image_processor(images, return_tensors="pt", **output_kwargs["images_kwargs"])
            num_tiles = image_features.pop("num_tiles")
            vision_inputs.update(**image_features)

        # constants
        max_len = self.tokenizer.model_max_length
        image_token_id = self.config.image_token_index
        
        input_ids = []
        labels = []
        
        # some parsing
        images: List[List[PIL.Image.Image]] = [instance["images"] for instance in instances]
        system_prompts: List[Union[str, None]] = [instance["system_prompt"] for instance in instances]
        conversations: List[List] = [instance["conversations"] for instance in instances]
        
        for system_prompt, cur_images, cur_convs in zip(system_prompts, images, conversations):
            cur_num_images = 0
            cur_input_ids = []
            cur_labels = []
            
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

            # a dirty hack to include eos token as part of the labels
            cur_assistant_masks[0, -1] = True
            
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
        
        return dict(
            **vision_inputs,
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.PAD_TOKEN_ID),
        )