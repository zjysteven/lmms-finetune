import re
from typing import Dict, List, Sequence, Union

import torch

from . import register_collator
from .base import BaseDataCollator


@register_collator("qwen-vl")
class QwenVLDataCollator(BaseDataCollator):
    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.eod_id # qwen-vl uses eod_id as pad_token_id

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # images
        # for qwen-vl these are filepaths to the images
        image_paths: List[List] = [instance["images"] for instance in instances]

        # texts
        # the dataset implementation assume conversations are [user, assistant, user, assistant, ...]
        system_prompts: List[Union[str, None]] = [instance["system_prompt"] for instance in instances]
        conversations: List[List] = [instance["conversations"] for instance in instances]
        input_ids = []
        labels = []

        im_start: int = self.tokenizer.im_start_id
        im_end: int = self.tokenizer.im_end_id
        nl_tokens: List = self.tokenizer("\n").input_ids
        _system: List = self.tokenizer("system").input_ids + nl_tokens
        max_len = self.tokenizer.model_max_length
        
        for cur_image_paths, system_prompt, cur_convs in zip(image_paths, system_prompts, conversations):
            cur_num_images = len(cur_image_paths)
            cur_image_idx = 0

            cur_input_ids = []
            cur_labels = []
            
            if system_prompt is not None:
                system = [im_start] + _system + self.tokenizer(system_prompt).input_ids + [im_end] + nl_tokens
                cur_input_ids.extend(system)
                cur_labels.extend([im_start] + [self.IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens)
            assert len(cur_input_ids) == len(cur_labels), "Input and label shapes do not match"
            
            for i, text in enumerate(cur_convs):
                # deal with the special image token format of qwen-vl
                image_token_start_locations = [m.start() for m in re.finditer('<image>', text)]
                for image_token_start in image_token_start_locations:
                    assert cur_image_idx < cur_num_images, "Image index out of bounds"
                    text = text[:image_token_start] + \
                        f"Picture {cur_image_idx + 1}: <img>{cur_image_paths[cur_image_idx]}</img>\n" + \
                        text[image_token_start + len("<image>"):]
                    cur_image_idx += 1      

                role = "<|im_start|>user" if i % 2 == 0 else "<|im_start|>assistant"
                _input_id = self.tokenizer(role).input_ids + nl_tokens + \
                    self.tokenizer(text).input_ids + [im_end] + nl_tokens
                cur_input_ids.extend(_input_id)

                if role == '<|im_start|>user':
                    _label = [im_start] + [self.IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
                elif role == '<|im_start|>assistant':
                    _label = [im_start] + [self.IGNORE_TOKEN_ID] * len(self.tokenizer(role).input_ids) + \
                        _input_id[len(self.tokenizer(role).input_ids) + 1:-2] + [im_end] + nl_tokens
                cur_labels.extend(_label)
            
            assert cur_image_idx == cur_num_images, "Not all images were used"

            assert len(cur_input_ids) == len(cur_labels), "Input and label shapes do not match"
            if max_len > len(cur_input_ids):
                cur_input_ids += [self.PAD_TOKEN_ID] * (max_len - len(cur_input_ids))
            if max_len > len(cur_labels):
                cur_labels += [self.IGNORE_TOKEN_ID] * (max_len - len(cur_labels))
            cur_input_ids = cur_input_ids[:max_len]
            cur_labels = cur_labels[:max_len]
            assert len(cur_input_ids) == len(cur_labels), "Input and label shapes do not match"

            input_ids.append(cur_input_ids[:])
            labels.append(cur_labels[:])

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.PAD_TOKEN_ID),
        )