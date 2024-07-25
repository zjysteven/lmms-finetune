import re
from typing import Dict, List, Sequence, Union
import PIL

import numpy as np
import torch

from . import register_collator
from .base import BaseDataCollator


@register_collator("llava-interleave")
class LLaVAInterleaveDataCollator(BaseDataCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # images
        vision_inputs = dict()
        flattened_images: List[PIL.Image.Image] = [x for instance in instances for x in instance["images"]]
        if len(flattened_images) > 0:
            vision_inputs.update(**self.processor.image_processor(flattened_images, return_tensors="pt"))
        num_images: List[int] = [len(instance["images"]) for instance in instances]

        # texts
        # the dataset implementation assume conversations are [user, assistant, user, assistant, ...]
        system_prompts: List[Union[str, None]] = [instance["system_prompt"] for instance in instances]
        conversations: List[List] = [instance["conversations"] for instance in instances]
        max_len = self.tokenizer.model_max_length

        user_token_id = self.tokenizer(
            "<|im_start|>user\n", add_special_tokens=False, padding=False, return_tensors="pt"
        )["input_ids"]
        assistant_token_id = self.tokenizer(
            "<|im_start|>assistant\n", add_special_tokens=False, padding=False, return_tensors="pt"
        )["input_ids"]

        total_image_tokens = 0
        input_ids = []
        labels = []
        
        for cur_num_images, system_prompt, cur_convs in zip(num_images, system_prompts, conversations):
            cur_num_image_tokens = 0
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
                    num_image_tokens = len([m.start() for m in re.finditer("<image>", text)])
                    cur_num_image_tokens += num_image_tokens
                    total_image_tokens += num_image_tokens

                    # .strip(): whitespaces and newlines are handled by chat_template
                    text = text.replace("<image>", "").strip()

                    cur_text.append({
                        "role": "user",
                        "content": [{"type": "text", "text": text}] + \
                            [{"type": "image"}] * num_image_tokens
                    })
                else:
                    cur_text.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": text}]
                    })
                
            assert cur_num_image_tokens == cur_num_images, "Not all images were used"
            
            cur_text = self.processor.apply_chat_template(cur_text, tokenize=False, add_generation_prompt=False)
            # a dirty tweak to avoid the double \n encoded after the assistant token
            # during training; otherwise there will be inconsistency between training
            # and inference where during inference only a single \n is encoded
            # this will also help locate the assistant token
            cur_text = cur_text.replace("<|im_start|>assistant\n\n", "<|im_start|>assistant\n")
            
            cur_input_ids = self.tokenizer(
                cur_text, truncation=True, max_length=max_len, return_tensors="pt"
            )["input_ids"]
            cur_labels = cur_input_ids.clone()

            if self.mask_question_tokens:
                # locate each question slice of the conversation
                # by finding the indices of the user and assistant tokens
                user_locs = np.where(np.array(cur_input_ids[0]) == user_token_id[0][0].item())[0]
                assistant_locs = np.where(np.array(cur_input_ids[0]) == assistant_token_id[0][0].item())[0]

                # filter potential false positives
                user_locs_filtered = [
                    i for i in user_locs if cur_input_ids[0, i:i + user_token_id.shape[1]].tolist() == user_token_id[0].tolist()
                ]
                assistant_locs_filtered = [
                    i for i in assistant_locs if cur_input_ids[0, i:i + assistant_token_id.shape[1]].tolist() == assistant_token_id[0].tolist()
                ]
                assert len(user_locs_filtered) == len(assistant_locs_filtered), "Number of user and assistant tokens do not match"

                start_inds = []; end_inds = []
                for i, (user_loc, assistant_loc) in enumerate(zip(user_locs_filtered, assistant_locs_filtered)):
                    start_inds.append(user_loc if i > 0 else 0)
                    end_inds.append(assistant_loc + assistant_token_id.shape[1])

                for start_ind, end_ind in zip(start_inds, end_inds):
                    cur_labels[0, start_ind:end_ind] = self.IGNORE_TOKEN_ID
            
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

        # sanity check
        assert total_image_tokens == len(flattened_images), "Number of image tokens does not match the number of images"

        input_ids = torch.cat(input_ids)
        labels = torch.cat(labels)

        return dict(
            **vision_inputs,
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.PAD_TOKEN_ID),
        )