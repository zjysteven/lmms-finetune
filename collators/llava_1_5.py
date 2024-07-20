import re
from typing import Dict, List, Sequence, Union
import PIL

import torch

from . import register_collator
from .base import BaseDataCollator


@register_collator("llava-1.5")
class LLaVA15DataCollator(BaseDataCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # images
        vision_inputs = dict()
        images: List[PIL.Image.Image] = [x for instance in instances for x in instance["images"]]
        if len(images) > 0:
            vision_inputs.update(**self.processor.image_processor(images, return_tensors="pt"))

        # texts
        # the dataset implementation assume conversations are [user, assistant, user, assistant, ...]
        system_prompts: List[Union[str, None]] = [instance["system_prompt"] for instance in instances]
        conversations: List[List] = [instance["conversations"] for instance in instances]
        raw_texts = []
        input_ids = []
        labels = []
        max_len = self.tokenizer.model_max_length

        total_image_tokens = 0
        
        for system_prompt, cur_convs in zip(system_prompts, conversations):
            cur_input_ids = []
            cur_labels = []
            cur_raw_texts = []
            
            if system_prompt is not None:
                system = self.tokenizer(system_prompt + "\n").input_ids
                cur_input_ids.extend(system)
                cur_labels.extend([self.IGNORE_TOKEN_ID] * len(system))
                cur_raw_texts.append(system_prompt + "\n")
            
            for i, text in enumerate(cur_convs):
                # shouldn't be a big deal?
                # but huggingface's chat template indicates that all
                # image tokens should be placed at the beginning of the text
                # this is also what the official implementation did in training
                num_image_tokens = len([m.start() for m in re.finditer("<image>", text)])
                total_image_tokens += num_image_tokens
                text = "<image>" * num_image_tokens + "\n" + text.replace("<image>", "")

                if i == 0:
                    _input_ids = self.tokenizer(
                        "USER: " + text + "\nASSISTANT: ",
                        add_special_tokens=system_prompt is None
                    ).input_ids
                    cur_input_ids.extend(_input_ids)
                    cur_labels.extend([self.IGNORE_TOKEN_ID] * len(_input_ids))
                    cur_raw_texts.append("USER: " + text + "\nASSISTANT: ")
                else:
                    if i % 2 == 0:
                        _input_ids = self.tokenizer(
                            "USER: " + text + "\nASSISTANT: ",
                            add_special_tokens=False
                        ).input_ids
                        cur_input_ids.extend(_input_ids)
                        cur_labels.extend([self.IGNORE_TOKEN_ID] * len(_input_ids))
                        cur_raw_texts.append("USER: " + text + "\nASSISTANT: ")
                    else:
                        _input_ids = self.tokenizer(
                            text + "\n" if i < len(cur_convs) - 1 else text,
                            add_special_tokens=False
                        ).input_ids
                        cur_input_ids.extend(_input_ids)
                        cur_labels.extend(_input_ids)
                        cur_raw_texts.append(text + "\n" if i < len(cur_convs) - 1 else text)
            
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
            raw_texts.append(cur_raw_texts[:])
        
        # sanity check
        assert total_image_tokens == len(images), "Number of image tokens does not match the number of images"

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        raw_texts = ["".join(x) for x in raw_texts]

        return dict(
            **vision_inputs,
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.PAD_TOKEN_ID),
            # raw_texts=raw_texts # for debugging
        )