import re
from typing import Dict, List, Sequence, Union
import PIL

import torch
from transformers.feature_extraction_utils import BatchFeature

from . import register_collator
from .base import BaseDataCollator


link = "https://github.com/microsoft/Phi-3CookBook/blob/5de3fe11f332109a111b2a4e3dfff467b31af7d7/code/04.Finetuning/vision_finetuning/finetune_hf_trainer_docvqa.py#L211"

@register_collator("phi3-v")
class Phi3VDataCollator(BaseDataCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        assert len(instances) == 1, f"Phi3-V only supports batch size of 1. See {link}."

        # images
        images: List[List] = [instance["images"] for instance in instances]
        assert len(images[0]) in [0, 1], \
            "Phi3-V only supports a single image per instance, " + \
            "as it is unclear how to handle the size differences between images."

        # texts
        # the dataset implementation assume conversations are [user, assistant, user, assistant, ...]
        system_prompts: List[Union[str, None]] = [instance["system_prompt"] for instance in instances]
        conversations: List[List] = [instance["conversations"] for instance in instances]

        input_ids = []
        labels = []
        max_len = self.tokenizer.model_max_length
        total_image_tokens = 0

        for system_prompt, cur_convs, cur_images in zip(system_prompts, conversations, images):
            cur_input_ids = []
            cur_labels = []
            
            system = ''
            if system_prompt is not None:
                system_message = {
                    "role": "system",
                    "content": system_prompt
                }
                system: str = self.tokenizer.apply_chat_template(
                    [system_message], tokenize=False, add_generation_prompt=False
                )

            cur_image_tokens = 0
            for i, text in enumerate(cur_convs):
                if i % 2 == 0:
                    # move all image tokens to the beginning
                    num_image_tokens = len([m.start() for m in re.finditer("<image>", text)])
                    new_image_tokens = "".join([f"<|image_{image_id + 1}|>" for image_id in range(cur_image_tokens, cur_image_tokens + num_image_tokens)])
                    if new_image_tokens:
                        new_image_tokens += "\n"
                    
                    total_image_tokens += num_image_tokens
                    text = new_image_tokens + text.replace("<image>", "")

                    prompt_message = {
                        'role': 'user',
                        'content': text
                    }
                    prompt = self.tokenizer.apply_chat_template(
                        [prompt_message], tokenize=False, add_generation_prompt=True
                    )
                    if i == 0:
                        prompt = system + prompt
                    
                    temp = self.processor(
                        prompt, 
                        cur_images[cur_image_tokens:cur_image_tokens + num_image_tokens] \
                            if num_image_tokens > 0 else None,
                    )
                    _input_ids = temp["input_ids"][0].tolist()
                    # definitely not the best way to do this
                    # but since we assume batch size of 1 and only 1 image
                    # we can just do this
                    if num_image_tokens > 0:
                        pixel_values = temp["pixel_values"]
                        image_sizes = temp["image_sizes"]

                    cur_input_ids.extend(_input_ids)
                    cur_labels.extend([self.IGNORE_TOKEN_ID] * len(_input_ids))

                    cur_image_tokens += num_image_tokens
                else:
                    answer = f'{text}<|end|>\n<|endoftext|>'
                    _input_ids = self.tokenizer(
                        answer, add_special_tokens=False
                    )["input_ids"]
                    cur_input_ids.extend(_input_ids)
                    cur_labels.extend(_input_ids)

            # sanity check
            assert cur_image_tokens == len(cur_images), "Number of image tokens does not match the number of images"
            
            assert len(cur_input_ids) == len(cur_labels), "Input and label shapes do not match"
            # since batch size is 1, we don't need to pad
            # if max_len > len(cur_input_ids):
            #     cur_input_ids += [self.PAD_TOKEN_ID] * (max_len - len(cur_input_ids))                
            # if max_len > len(cur_labels):
            #     cur_labels += [self.IGNORE_TOKEN_ID] * (max_len - len(cur_labels))
            # cur_input_ids = cur_input_ids[:max_len]
            # cur_labels = cur_labels[:max_len]
            # assert len(cur_input_ids) == len(cur_labels), "Input and label shapes do not match"

            input_ids.append(cur_input_ids[:])
            labels.append(cur_labels[:])
        
        # sanity check
        assert total_image_tokens == len(images), "Number of image tokens does not match the number of images"

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return dict(
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.PAD_TOKEN_ID),
        )