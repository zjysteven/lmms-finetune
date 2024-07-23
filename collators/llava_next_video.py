import re
from typing import Dict, List, Sequence, Union
import PIL

import numpy as np
import torch

from . import register_collator
from .base import BaseDataCollator


@register_collator("llava-next-video")
class LLaVANeXTVideoDataCollator(BaseDataCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        vision_inputs = dict()

        # images
        images: List[PIL.Image.Image] = [x for instance in instances for x in instance["images"]]
        if len(images) > 0:
            vision_inputs.update(**self.processor.image_processor(images, return_tensors="pt"))

        # videos
        # we do manual padding here so that each video has the same length
        # so that we can batch them together
        # here has a limitation: huggingface implementation
        # currently consumes all batched frames (no "unpadding" is ever done)
        # so if there are clips that are significantly shorter than the longest clip
        # then the model will be trained on many padded frames
        videos: List[np.ndarray] = [x for instance in instances for x in instance["videos"]]
        if len(videos) > 0:
            max_num_frames = max([x.shape[0] for x in videos])
            for i, video in enumerate(videos):
                if video.shape[0] < max_num_frames:
                    # pad = np.array(PIL.Image.new("RGB", video.shape[1:3][::-1], color=(0, 0, 0)))
                    pad = np.zeros((video.shape[0], video.shape[1], video.shape[2]), dtype=np.uint8)
                    pad = np.expand_dims(pad, axis=0).repeat(max_num_frames - video.shape[0], axis=0)
                    videos[i] = np.concatenate([video, pad], axis=0)
            vision_inputs.update(**self.processor.video_processor(videos, return_tensors="pt"))

        # texts
        # the dataset implementation assume conversations are [user, assistant, user, assistant, ...]
        system_prompts: List[Union[str, None]] = [instance["system_prompt"] for instance in instances]
        conversations: List[List] = [instance["conversations"] for instance in instances]
        input_ids = []
        labels = []
        max_len = self.tokenizer.model_max_length

        total_image_tokens = 0
        total_video_tokens = 0
        
        for system_prompt, cur_convs in zip(system_prompts, conversations):
            cur_input_ids = []
            cur_labels = []
            
            if system_prompt is not None:
                system = self.tokenizer(system_prompt + "\n").input_ids
                cur_input_ids.extend(system)
                cur_labels.extend([self.IGNORE_TOKEN_ID] * len(system))
            
            for i, text in enumerate(cur_convs):
                # shouldn't be a big deal?
                # but huggingface's chat template indicates that all
                # image/video tokens should be placed at the beginning of the text
                num_image_tokens = len([m.start() for m in re.finditer("<image>", text)])
                total_image_tokens += num_image_tokens

                num_video_tokens = len([m.start() for m in re.finditer("<video>", text)])
                total_video_tokens += num_video_tokens

                text = "<image>" * num_image_tokens + "\n" \
                    + "<video>" * num_video_tokens + "\n" \
                    + text.replace("<image>", "").replace("<video>", "")

                if i == 0:
                    _input_ids = self.tokenizer(
                        "USER: " + text + "\nASSISTANT: ",
                        add_special_tokens=system_prompt is None
                    ).input_ids
                    cur_input_ids.extend(_input_ids)
                    cur_labels.extend([self.IGNORE_TOKEN_ID] * len(_input_ids))
                else:
                    if i % 2 == 0:
                        _input_ids = self.tokenizer(
                            "USER: " + text + "\nASSISTANT: ",
                            add_special_tokens=False
                        ).input_ids
                        cur_input_ids.extend(_input_ids)
                        cur_labels.extend([self.IGNORE_TOKEN_ID] * len(_input_ids))
                    else:
                        _input_ids = self.tokenizer(
                            text + "\n" if i < len(cur_convs) - 1 else text,
                            add_special_tokens=False
                        ).input_ids
                        cur_input_ids.extend(_input_ids)
                        cur_labels.extend(_input_ids)
            
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

        # sanity check
        assert total_image_tokens == len(images), "Number of image tokens does not match the number of images"
        assert total_video_tokens == len(videos), "Number of video tokens does not match the number of videos"

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return dict(
            **vision_inputs,
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.PAD_TOKEN_ID),
        )